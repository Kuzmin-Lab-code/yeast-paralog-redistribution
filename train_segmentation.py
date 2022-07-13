import os

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm.auto import tqdm

from modules.segmentation import dataset, model
from modules.tools import util
from modules.tools.transforms import get_train_transforms
from modules.tools.types import *
from modules.tools.util import SubsetSampler


@hydra.main(config_path="config", config_name="segmentation.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    cwd = Path(get_original_cwd())
    util.fix_seed(cfg.seed)

    print(
        f"Run experiment {cfg.name} from {cwd}\nWork in {os.getcwd()}\nSeed={cfg.seed}"
    )

    # Dataset
    ads = dataset.AnnotatedDataset(
        path=cwd / cfg.data.path,
        metainfo=cwd / cfg.data.metainfo,
        log=cfg.data.log,
        std=cfg.data.std,
        distance_transform=cfg.data.distance_transform,
        transforms=get_train_transforms(cfg.data.crop),
    )

    # Make stratified validation split (1 image per gene, 250 in total)
    indices = np.arange(len(ads))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=cfg.training.valid_split,
        random_state=cfg.seed,
        shuffle=True,
        stratify=ads.metainfo.gene.tolist(),
    )

    print(
        f"train {len(train_idx)}, valid {len(valid_idx)} (unique genes {ads.metainfo.gene[valid_idx].nunique()})"
    )

    train_loader = DataLoader(
        ads,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        ads,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        sampler=SubsetSampler(valid_idx),
        pin_memory=True,
        drop_last=False,
    )

    wandb_logger = None
    if not cfg.dev:
        wandb_logger = WandbLogger(project=cfg.name)
        wandb_logger.experiment.config.update(util.flatten_cfg(cfg))
    tensorboard_logger = TensorBoardLogger(save_dir="tensorboard", name=cfg.name)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.monitor, mode="max", save_last=True, save_top_k=3
    )

    mdl = model.SegmentationModel(
        **cfg.model,
        lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        epochs=cfg.training.epochs,
        classes_aux=ads.n_classes,
        wandb_logger=wandb_logger,
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=cfg.training.epochs,
        logger=[wandb_logger, tensorboard_logger] if not cfg.dev else None,
        callbacks=[checkpoint_callback],
        fast_dev_run=cfg.dev,
    )

    trainer.fit(mdl, train_loader, valid_loader)
    return


if __name__ == "__main__":
    main()
