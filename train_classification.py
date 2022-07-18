import os

import albumentations as A
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from modules.classification import dataset, model, network
from modules.tools import metrics, util
from modules.tools.types import *


@hydra.main(config_path="config", config_name="classification.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    cwd = Path(get_original_cwd())
    util.fix_seed(cfg.seed)
    print(f"Run from {cwd.as_posix()}, cwd {os.getcwd()}")

    transforms = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(),
            # A.RandomBrightnessContrast(),
            # A.RandomGamma(),
            # A.Blur(),
            # A.GaussNoise(var_limit=0.01),
        ]
    )

    wt_dataset = dataset.FramesDataset(
        path_data=cwd / cfg.data.path_data,
        dir_frames=cfg.data.dir_frames,
        select="wt",
        validation_field=cfg.data.validation_field,
        transforms=transforms,
        seed=cfg.seed,
        normalize=cfg.data.normalize,
    )

    train_sampler, valid_sampler = wt_dataset.get_train_valid_samplers()
    train_loader = DataLoader(
        wt_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        wt_dataset,
        batch_size=cfg.training.batch_size,
        sampler=valid_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    mdl = model.get_model(cfg, n_classes=wt_dataset.n_classes)

    # Loggers
    logger = None
    if not cfg.dev:
        wandb_logger = WandbLogger(project=cfg.name)
        wandb_logger.experiment.config.update(util.flatten_cfg(cfg))
        tensorboard_logger = TensorBoardLogger(save_dir="tensorboard", name=cfg.name)
        logger = [wandb_logger, tensorboard_logger]

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=3)
    ]

    if cfg.model.only_head > 0:
        print(f"Finetune the head for {cfg.model.only_head} epochs")
        callbacks.append(
            BackboneFinetuning(
                unfreeze_backbone_at_epoch=cfg.model.only_head,
                backbone_initial_ratio_lr=1,
                # verbose=True,
                train_bn=True,
            )
        )

    # Training
    trainer = Trainer(
        gpus=1,
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=cfg.dev,
    )
    trainer.fit(mdl, train_loader, valid_loader)

    # Inference
    predictions, features, y = mdl.inference(valid_loader)

    top_k = [1, 5, 10]
    accuracy = {f"acc@top{k}": metrics.accuracy(y, predictions, top=k) for k in top_k}
    for k, acc in accuracy.items():
        print(f"{k}: {acc * 100: .2f}")

    if not cfg.dev:
        wandb_logger.log_metrics(accuracy)


if __name__ == "__main__":
    main()
