import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from torch import nn
from tqdm.auto import tqdm

from modules.classification.loss import ArcFaceLoss, ArcMarginProductPlain
from modules.classification.network import resnet18


class LitModel(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module = resnet18(n_classes=182),
        criterion: nn.Module = nn.CrossEntropyLoss(),
        metric_criterion: nn.Module = ArcFaceLoss(),
        metric_coefficient: float = 0.25,
        scale_factor: float = 1.0,
        seed: int = 15,
    ):
        super().__init__()
        np.random.seed(seed)

        self.network = network
        self.criterion = criterion
        self.metric_criterion = metric_criterion
        self.metric = ArcMarginProductPlain(
            self.network.fc.in_features, self.network.fc.out_features
        )
        self.metric_coefficient = metric_coefficient
        # Metrics
        self.accuracy = Accuracy(num_classes=self.network.n_classes)
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.scale_factor != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=self.scale_factor, mode="bicubic"
            )

        features = self.network.features(x)
        out = self.network.fc(features)
        metric_out = self.metric(features)
        clf_loss = self.criterion(out, y)

        if self.metric_coefficient > 0:
            metric_loss = self.metric_criterion(metric_out, y)

            loss = (
                self.metric_coefficient * metric_loss
                + (1 - self.metric_coefficient) * clf_loss
            )

            result = pl.TrainResult(minimize=loss)
            result.log("clf_loss", clf_loss)
            result.log("metric_loss", metric_loss)
        else:
            loss = clf_loss
            result = pl.TrainResult(minimize=loss)

        acc = self.accuracy(out.argmax(1), y)
        result.log("train_loss", loss)
        result.log("train_acc", acc)
        #         result.log('progress_bar', {'acc': acc})

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

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict(
            {"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True
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

    def inference(self, loader):
        self.freeze()
        self.network = self.network.cuda()
        self.eval()
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

                fts = self.network.features(x.cuda())
                out = self.network.fc(fts)

                acc.append(self.accuracy(out.argmax(1), y.cuda()))
                iterator.set_postfix({"acc": np.mean(acc) * 100})

                predictions.append(out.cpu().numpy())
                features.append(fts.cpu().numpy())
                ys.append(y.cpu().numpy())

        predictions = np.concatenate(predictions)
        features = np.concatenate(features)
        ys = np.concatenate(ys)

        self.unfreeze()
        return predictions, features, ys
