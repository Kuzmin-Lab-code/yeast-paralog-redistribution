import itertools
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from modules.classification import network
from modules.classification.loss import ArcFaceLoss, ArcMarginProductPlain
from modules.tools import util


class LitModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        metric_criterion: nn.Module = ArcFaceLoss(),
        metric_coefficient: float = 0.25,
        scale_factor: float = 1.0,
        seed: int = 15,
        epochs: int = 200,
        lr: float = 3e-4,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        np.random.seed(seed)

        self.backbone = backbone
        self.head = head
        self.criterion = criterion

        self.metric_coefficient = metric_coefficient
        if self.metric_coefficient > 0:
            self.metric_criterion = metric_criterion
            self.metric = ArcMarginProductPlain(
                self.head.in_features, self.head.out_features
            )
        # Metrics
        self.accuracy = Accuracy()
        self.scale_factor = scale_factor

        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr

    def forward(self, x):
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.scale_factor != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=self.scale_factor, mode="bicubic"
            )

        features = self.backbone(x)
        out = self.head(features)
        clf_loss = self.criterion(out, y)

        result = {}
        if self.metric_coefficient > 0:
            metric_out = self.metric(features)
            metric_loss = self.metric_criterion(metric_out, y)

            loss = (
                self.metric_coefficient * metric_loss
                + (1 - self.metric_coefficient) * clf_loss
            )

            result["clf_loss"] = clf_loss
            result["metric_loss"] = metric_loss
        else:
            loss = clf_loss

        acc = self.accuracy(out.argmax(1), y)
        result["loss"] = loss
        result["acc"] = acc
        self.log_dict(result, prog_bar=True, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.scale_factor != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=self.scale_factor, mode="bicubic"
            )

        out = self(x)
        loss = self.criterion(out, y)
        acc = self.accuracy(out.argmax(1), y)

        result = {"val_loss": loss, "val_acc": acc}
        self.log_dict(result, prog_bar=True, on_epoch=True)

        return result

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                itertools.chain(self.backbone.parameters(), self.head.parameters()),
                lr=self.lr,
            )
        ]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[0], T_max=self.epochs, eta_min=self.min_lr
        )
        schedulers = [
            {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     optimizers[0], factor=0.1
                # ),
                "scheduler": scheduler,
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizers, schedulers

    def on_validation_epoch_start(self):
        for dl in self.trainer.val_dataloaders:
            try:
                dl.dataset.eval()
            except AttributeError:
                dl.dataset.dataset.eval()

    def on_train_epoch_start(self):
        try:
            # self.trainer.train_dataloader.dataset.train()
            self.trainer.train_dataloader.loaders.dataset.train()
        except AttributeError:
            # self.trainer.train_dataloader.dataset.dataset.train()
            self.trainer.train_dataloader.loaders.dataset.dataset.train()

    def inference(self, loader):
        self.freeze()
        self.backbone = self.backbone.cuda()
        self.head = self.head.cuda()
        loader.dataset.eval()

        ys = []
        predictions = []
        features = []
        acc = []

        with torch.no_grad():
            iterator = tqdm(loader)
            for x, y in iterator:

                if self.scale_factor != 1:
                    x = torch.nn.functional.interpolate(
                        x, scale_factor=self.scale_factor, mode="bicubic"
                    )

                fts = self.backbone(x.cuda())
                out = self.head(fts)

                acc.append(self.accuracy(out.argmax(1).cpu(), y).cpu().numpy())
                iterator.set_postfix({"acc": np.mean(acc) * 100})

                predictions.append(out.cpu().numpy())
                features.append(fts.cpu().numpy())
                ys.append(y.cpu().numpy())

        predictions = np.concatenate(predictions)
        features = np.concatenate(features)
        ys = np.concatenate(ys)

        self.unfreeze()
        return predictions, features, ys


def get_model(cfg: DictConfig, cwd: Union[str, Path], n_classes: int) -> LitModel:
    cwd = Path(cwd)
    checkpoint_path = getattr(cfg, "checkpoint", None)

    if checkpoint_path is not None:
        # TODO add option to create encoder model without checkpoint
        print("Load encoder from segmentation checkpoint")
        cfg_segmentation, weights = util.load_cfg_and_checkpoint(cwd / checkpoint_path)
        print(f"Create a {cfg_segmentation.model.encoder_name} model")
        backbone = network.EncoderHeadless(
            encoder_name=cfg_segmentation.model.encoder_name,
            in_channels=cfg_segmentation.model.in_channels,  # assuming the same for classification
            dropout=cfg.model.dropout,
        )
        backbone.load_state_dict_from_segmentation(weights)
    else:
        # TODO parametrize with torchvision/timm models
        print("Create a resnet18 model")
        backbone = network.ResidualNetworkHeadless(
            num_units=2,  # resnet18
            in_channels=1,
            base_channels=cfg.model.base_channels,
            dropout=cfg.model.dropout,
        )

    head = nn.Linear(in_features=backbone.out_channels, out_features=n_classes)

    return LitModel(
        backbone=backbone,
        head=head,
        scale_factor=cfg.model.scale_factor,
        seed=cfg.seed,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        metric_coefficient=cfg.model.metric_coefficient,
    )
