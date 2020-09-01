import numpy as np
import pytorch_lightning as pl
import torch
from network import resnet18
from pytorch_lightning.metrics import Accuracy
from torch import nn


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module = resnet18(n_classes=182),
        criterion: nn.Module = nn.CrossEntropyLoss(),
        seed: int = 15,
    ):
        super().__init__()
        np.random.seed(seed)

        self.model = model
        self.criterion = criterion

        # Metrics
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        loss = self.criterion(out, y)
        acc = self.accuracy(out.argmax(1), y)

        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        result.log("train_acc", acc)
        #         result.log('progress_bar', {'acc': acc})

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        loss = self.criterion(out, y)
        acc = self.accuracy(out.argmax(1), y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        result.log("val_acc", acc)

        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)
