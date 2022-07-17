import os

import albumentations as A
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
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

    checkpoint_path = getattr(cfg, "checkpoint", None)
    if checkpoint_path is not None:
        print("Load encoder from segmentation checkpoint")
        cfg_segmentation, weights = util.load_cfg_and_checkpoint(cwd / checkpoint_path)
        print(f"Create a {cfg_segmentation.model.encoder_name} model")
        net = network.EncoderWithHead(
            encoder_name=cfg_segmentation.model.encoder_name,
            in_channels=cfg_segmentation.model.in_channels,  # assuming the same for classification
            n_classes=wt_dataset.n_classes,
        )
        net.load_state_dict_from_segmentation(weights)
    else:
        # todo parametrize with torchvision/timm models
        print("Create a resnet18 model")
        net = network.resnet18(
            n_classes=wt_dataset.n_classes, base_channels=cfg.model.base_channels
        )

    mdl = model.LitModel(
        network=net,
        scale_factor=cfg.model.scale_factor,
        seed=cfg.seed,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        metric_coefficient=cfg.model.metric_coefficient,
    )

    wandb_logger = WandbLogger(project=cfg.name)
    wandb_logger.experiment.config.update(util.flatten_cfg(cfg))
    tensorboard_logger = TensorBoardLogger(save_dir="tensorboard", name=cfg.name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", mode="max", save_last=True, save_top_k=3
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=cfg.training.epochs,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[checkpoint_callback],
        fast_dev_run=cfg.dev,
    )
    trainer.fit(mdl, train_loader, valid_loader)

    predictions, features, y = mdl.inference(valid_loader)

    top_k = [1, 5, 10]
    accuracy = {f"acc@top{k}": metrics.accuracy(y, predictions, top=k) for k in top_k}
    for k, acc in accuracy.items():
        print(f"{k}: {acc * 100: .2f}")
    wandb_logger.log_metrics(accuracy)


if __name__ == "__main__":
    main()
