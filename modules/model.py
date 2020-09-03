from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from network import resnet18
from pytorch_lightning.metrics import Accuracy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, ks_2samp, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from torch import nn
from tqdm.auto import tqdm

Array = Union[np.ndarray]


class LitModel(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module = resnet18(n_classes=182),
        criterion: nn.Module = nn.CrossEntropyLoss(),
        seed: int = 15,
    ):
        super().__init__()
        np.random.seed(seed)

        self.network = network
        self.criterion = criterion

        # Metrics
        # TODO infer from data
        self.accuracy = Accuracy(num_classes=self.network.n_classes)

    def forward(self, x):
        return self.network(x)

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

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     features = self.model.network(x)
    #     out = self.model.fc(features)
    #     loss = self.criterion(out, y)
    #
    #     return {"loss": loss.unsqueeze(0), "features": features, "pred": out, "y": y}
    #
    # def test_end(self, outputs):
    #     """
    #     Aggregate test loss and predictions
    #     """
    #     stacked = {k: torch.cat([x[k] for x in outputs], 0) for k in outputs[0].keys()}
    #     stacked["loss"] = stacked["loss"].mean()
    #     return stacked

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=3e-4)

    def on_validation_epoch_start(self):
        for dl in self.trainer.val_dataloaders:
            dl.dataset.eval()

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.dataset.train()

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


def aggregate(metainfo: pd.DataFrame, features: Array, group_by_replicate: bool = True):
    """
    Aggregate features by metainfo `label` column
    :param metainfo: metainfo about files
    :param features: extracted features
    :param group_by_replicate: extracted features
    :return:
    """
    assert (
        len(metainfo["label"].unique()) == 4
    ), "There should be 4 labels in metainfo for this analysis"

    features_df = pd.DataFrame(features)
    features_df["label"] = metainfo["label"]
    features_df["replicate"] = metainfo["replicate"]
    groups = ["replicate", "label"] if group_by_replicate else "label"

    return features_df.groupby(groups).mean()


def measure_pairwise_distance(
    aggregated_features: Array, pairs: List[str], metric: str = "cosine"
):
    """
    Measure distances between WT and DELTA conditions:
                         POR1-GFP POR2-WT 	POR2-GFP POR1-WT
    POR1-GFP POR2-DELTA 	0.047326 	        1.049365
    POR2-GFP POR1-DELTA 	0.811942        	1.097588
    :param aggregated_features: array of aggregated features (4, n_features)
    :param pairs: list of row names
    :param metric: metric to put in table
    :return: DataFrame, for both genes in pair measure distances
    """
    df = (
        pd.DataFrame(
            squareform(pdist(aggregated_features, metric=metric)),
            columns=pairs,
            index=pairs,
        )
        .drop([p for p in pairs if p.endswith("DELTA")], axis=1)
        .drop([p for p in pairs if p.endswith("WT")], axis=0)
    )
    return df


def stat_test_pc(features, labels, stat_test=kruskal, pc: int = 0, **kwargs):
    """Run univariate statistical test for the principal component"""
    tfm = PCA(n_components=1)
    tfm_features = tfm.fit_transform(features)
    feature = tfm_features[:, pc].squeeze()
    groups, group_counts = np.unique(labels, return_counts=True)

    # stat_test(delta, wt)
    res = {
        groups[0]
        .split("-")[0]: stat_test(
            feature[labels == groups[0]], feature[labels == groups[1]], **kwargs
        )
        ._asdict(),
        groups[2]
        .split("-")[0]: stat_test(
            feature[labels == groups[2]], feature[labels == groups[3]], **kwargs
        )
        ._asdict(),
    }
    res = pd.DataFrame(res).T
    res["test"] = stat_test.__name__

    return res, feature


def test_pairs(pairs: List):
    """
    Test all the pairs in format KIN1-KIN2
    :param pairs:
    :return:
    """
    for pair in pairs:
        g1, g2 = pair.split("-")
        dataset = FramesDataset()
