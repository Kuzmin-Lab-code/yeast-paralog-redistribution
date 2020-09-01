import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import transforms as T
from torch import Tensor
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler

# from torchvision import transforms as T


def prepare_metainfo_labels(metainfo: pd.DataFrame) -> pd.DataFrame:
    """
    Produces `label` and `class` columns for paralog pairs
        GFP 	natMX4 	pairs 	    label                   class
        KIN1 	NaN 	KIN1-KIN2 	KIN1-GFP KIN2-WT        0
        KIN1 	KIN2 	KIN1-KIN2 	KIN1-GFP KIN2-DELTA     0
    `label` made from `GFP` and `natMX4` data, `class` is numerical encoding of `GFP`
    :param metainfo: prepared metainfo dataframe with columns GFP,
    :return:
    """

    labels = []
    for i, r in metainfo.iterrows():
        # Handle control pairs
        if r["pairs"].startswith("control"):
            labels.append("control")
        else:
            # Get paralog by excluding GFP from pair
            pair = r["pairs"].split("-")
            pair.remove(r["GFP"])
            # Construct label
            label = f"{r['GFP']}-GFP {pair[0]}-{'DELTA' if isinstance(r['natMX4'], str) else 'WT'}"
            labels.append(label)
    # Assign labels
    metainfo["label"] = labels

    # Make classes
    metainfo["GFP"] = metainfo["GFP"].astype("category")
    metainfo["class"] = metainfo["GFP"].cat.codes.astype(np.uint32)

    return metainfo


class FramesDataset(Dataset):
    def __init__(
        self,
        replicate: Union[str, int] = "*",
        path: str = "./data/frames_separated/",
        fmt: str = "npy",
        validation_field: Union[int, None] = 4,
        select: Union[str, List[str]] = "wt",
        seed: int = 56,
        transforms: List[A.BasicTransform] = [],
    ):
        super().__init__()
        # Set random seed for potential split
        np.random.seed(seed)
        # Check if validation field is provided within correct boundaries
        assert validation_field in (None, 1, 2, 3, 4)
        assert str(replicate) in ("*", "1", "2", "3")

        # Store variables
        self.path = Path(path)
        self.replicate = replicate
        self.fmt = fmt
        self.validation_field = validation_field
        self.select = select

        # Read file list
        self.files = sorted(
            glob.glob(str(self.path / f"replicate{replicate}" / "*" / f"*.{fmt}"))
        )

        # Create default metainfo
        # >>> self.metainfo.head()
        #         URL     replicate       cell_id       file
        # 0     0000000       1              0       path/to/file
        self.metainfo = pd.DataFrame(
            [np.array(f.split("/"))[[-2, 2, -1]] for f in self.files],
            columns=["URL", "replicate", "cell_id"],
        )
        self.metainfo["file"] = self.files

        # Read metainfo about GFP tags
        metainfo_gfp = pd.read_csv("./data/metainfo.csv", index_col=0, dtype="str")

        # Select subset by parameters
        if select is None:
            pass
        elif select == "wt":
            metainfo_gfp = metainfo_gfp[metainfo_gfp.natMX4.isna()]
        else:
            metainfo_gfp = metainfo_gfp[metainfo_gfp.GFP.isin(select)]

        # Prepare labels and classes
        metainfo_gfp = prepare_metainfo_labels(metainfo_gfp)
        self.n_classes = metainfo_gfp.GFP.cat.codes.max() + 1

        # Merge info by URL
        self.metainfo = self.metainfo[self.metainfo.URL.isin(metainfo_gfp.URL.unique())]
        self.metainfo = self.metainfo.merge(metainfo_gfp, how="left", on="URL")

        # Hold out validation
        if validation_field is None:
            indices = list(range(len(self.metainfo)))
            np.random.shuffle(indices)
            n_val = 0.25 * len(indices)
            self.train_indices, self.valid_indices = indices[:-n_val], indices[-n_val:]
        else:
            from_validation_field = self.metainfo.Field == str(validation_field)
            self.train_indices = np.where(~from_validation_field)[0]
            self.valid_indices = np.where(from_validation_field)[0]
            np.random.shuffle(self.train_indices)

        self.transforms = A.Compose([*transforms, T.ToTensorV2()])

    def __getitem__(self, i) -> Tuple[Tensor, int]:
        f, cls = self.metainfo.iloc[i][["file", "class"]]
        img = np.load(f).astype(np.float32)
        cls = int(cls)

        # normalize each image
        # (intensities could vary largely)
        #         img = np.log(img + 1)
        img -= img.mean()
        img /= img.std()
        img = self.transforms(img)

        return img, cls

    def __len__(self) -> int:
        return len(self.files)

    def get_train_valid_samplers(self) -> Tuple[Sampler, Sampler]:
        """
        :return: random samplers from train and validation indices
        """
        train_sampler = SubsetRandomSampler(self.train_indices)
        valid_sampler = SubsetRandomSampler(self.valid_indices)
        return train_sampler, valid_sampler
