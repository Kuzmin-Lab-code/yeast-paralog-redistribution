from typing import *

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import Accuracy, Dice, F1Score, MetricCollection, Precision, Recall
from tqdm.auto import tqdm


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = "unet",
        encoder_name: str = "resnet34",
        in_channels: int = 1,
        classes: int = 1,
        encoder_weights: Optional[str] = None,
        activation: Optional[str] = "sigmoid",
        criterion: nn.Module = nn.BCELoss(),
        lambda_aux: float = 0,
        criterion_aux: nn.Module = nn.CrossEntropyLoss(),
        classes_aux: int = 250,
        lr: float = 3e-4,
        min_lr: float = 1e-7,
        epochs: int = 0,
        wandb_logger: Optional[WandbLogger] = None,
        **kwargs,
    ):
        super().__init__()

        aux_params = None
        if lambda_aux > 0:
            aux_params = dict(
                pooling="avg",
                dropout=0.25,  # todo parametrize
                classes=classes_aux,
            )

        self.network = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            encoder_weights=encoder_weights,
            activation=activation,
            aux_params=aux_params,
            **kwargs,
        )

        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.lambda_aux = lambda_aux
        self.wandb_logger = wandb_logger

        # Metrics
        metrics = MetricCollection(
            [
                Accuracy(average="samples"),
                Precision(average="samples"),
                Recall(average="samples"),
                F1Score(average="samples"),
                Dice(average="samples"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, x):
        self.network(x)

    def compute(self, batch):
        out = self.network(batch["image"])

        if self.lambda_aux > 0:
            out, out_aux = out

        losses = {"loss": self.criterion(out, batch["mask"])}

        if self.lambda_aux > 0:
            losses["loss_seg"] = losses["loss"]
            losses["loss_clf"] = self.criterion_aux(out_aux, batch["label"])
            losses["loss"] = losses["loss_seg"] + self.lambda_aux * losses["loss_clf"]

        return out, losses

    def training_step(self, batch, batch_idx):
        out, losses = self.compute(batch)
        metrics = self.train_metrics(out, (batch["mask"] > 0.5).int())
        losses.update(metrics)
        self.log_dict(losses, prog_bar=True, on_epoch=True)
        return losses

    def on_validation_epoch_start(self):
        for dl in self.trainer.val_dataloaders:
            try:
                dl.dataset.eval()
            except AttributeError:
                dl.dataset.dataset.eval()

    def on_train_epoch_start(self):
        try:
            self.trainer.train_dataloader.loaders.dataset.train()
        except AttributeError:
            self.trainer.train_dataloader.loaders.dataset.dataset.train()

    def validation_step(self, batch, batch_idx):
        out, losses = self.compute(batch)
        losses = {f"val_{k}": v for k, v in losses.items()}
        metrics = self.valid_metrics(out, (batch["mask"] > 0.5).int())
        losses.update(metrics)
        self.log_dict(losses, prog_bar=True, on_epoch=True)

        # log images
        if batch_idx == 0 and self.wandb_logger:
            self.wandb_logger.log_image(
                key="images",
                images=[
                    wandb.Image(batch["image"][0].cpu().numpy(), caption="input"),
                    wandb.Image(batch["mask"][0].cpu().numpy() > 0, caption="target"),
                    wandb.Image(out[0, 0].cpu().numpy(), caption="output"),
                ],
            )

        return losses

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(self.network.parameters(), lr=self.lr)]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[0], T_max=self.epochs, eta_min=self.min_lr
        )
        schedulers = [
            {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     optimizers[0], factor=0.1
                # ),
                "scheduler": scheduler,
                "monitor": "val_f1",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizers, schedulers

    def inference(self, loader, device="cuda"):
        self.freeze()
        self.eval()
        loader.dataset.eval()
        self.network = self.network.to(device)
        self.valid_metrics.reset()

        ys = []
        predictions = []

        with torch.no_grad():
            iterator = tqdm(loader)
            for batch in iterator:
                batch = {k: v.to(device) for k, v in batch.items()}
                # todo figure out why original shape is not set automatically
                loader.dataset.original_shape = batch["image"][0].shape

                out, results = self.compute(batch)
                out = loader.dataset.crop_to_original(out)
                mask = loader.dataset.crop_to_original(batch["mask"])

                metrics = self.valid_metrics(out, (mask > 0.5).int())
                metrics = {k: f"{v.cpu().numpy(): .4f}" for k, v in metrics.items()}
                iterator.set_postfix(metrics)

                predictions.append(out.cpu().numpy())
                ys.append(mask.cpu().numpy())

        predictions = np.concatenate(predictions)
        ys = np.concatenate(ys)

        self.unfreeze()
        return predictions, ys
