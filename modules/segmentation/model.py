from typing import *

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from segmentation_models_pytorch.utils import losses
from torch import nn
from tqdm.auto import tqdm


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 1,
        classes: int = 1,
        encoder_weights: Optional[str] = None,
        activation: Optional[str] = "sigmoid",
        **kwargs
    ):
        super().__init__()

        self.network = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            encoder_weights=encoder_weights,
            activation=activation,
            **kwargs
        )

        self.criterion = nn.BCELoss() if classes == 1 else nn.NLLLoss()
        self.accuracy = pl.metrics.Accuracy(num_classes=classes)

    def forward(self, x):
        self.network(x)

    def out_loss_acc(self, batch):
        x, y = batch
        out = self.network(x)
        loss = self.criterion(out, y)
        # TODO implement for multiclass?
        acc = self.accuracy(torch.round(out), torch.round(y))
        return out, loss, acc

    def training_step(self, batch, batch_idx):
        out, loss, acc = self.out_loss_acc(batch)

        result = pl.TrainResult(minimize=loss)
        result.log("loss", loss)
        result.log("acc", acc)

        return result

    def on_validation_epoch_start(self):
        for dl in self.trainer.val_dataloaders:
            try:
                dl.dataset.eval()
            except AttributeError:
                dl.dataset.dataset.eval()

    def on_train_epoch_start(self):
        try:
            self.trainer.train_dataloader.dataset.train()
        except AttributeError:
            self.trainer.train_dataloader.dataset.dataset.train()

    def validation_step(self, batch, batch_idx):
        out, loss, acc = self.out_loss_acc(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict(
            {"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True
        )

        # log images
        if batch_idx == 0:
            # using weights and biases
            if isinstance(self.logger, WandbLogger):
                # self.logger.experiment.log({
                #     'input': [wandb.Image(x) for x in batch[0].cpu().numpy()],
                #     'target': [wandb.Image(y) for y in batch[1].cpu().numpy()],
                #     'output': [wandb.Image(o) for o in out.cpu().numpy()],
                # })
                self.logger.experiment.log(
                    {
                        "images": [
                            wandb.Image(batch[0][0].cpu().numpy(), caption="input"),
                            wandb.Image(batch[1][0].cpu().numpy(), caption="target"),
                            wandb.Image(out[0].cpu().numpy(), caption="output"),
                        ],
                    }
                )

        return result

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(self.network.parameters(), lr=3e-4)]
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0], factor=0.1
                ),
                "monitor": "val_checkpoint_on",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizers, schedulers

    def inference(self, loader):
        self.freeze()
        self.eval()
        loader.dataset.eval()
        self.network = self.network.cuda()

        ys = []
        predictions = []
        accuracy = []

        with torch.no_grad():
            iterator = tqdm(loader)
            for x, y in iterator:
                out, loss, acc = self.out_loss_acc((x.cuda(), y.cuda()))
                accuracy.append(acc)
                iterator.set_postfix({"acc": np.mean(accuracy) * 100})

                predictions.append(loader.dataset.crop_to_original(out.cpu()).numpy())
                ys.append(loader.dataset.crop_to_original(y.cpu()).numpy())

        predictions = np.concatenate(predictions)
        ys = np.concatenate(ys)

        self.unfreeze()
        return predictions, ys

