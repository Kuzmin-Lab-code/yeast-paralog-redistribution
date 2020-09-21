import glob
from abc import ABC

import albumentations as A
import numpy as np
import pandas as pd
import torch.nn.functional as F
from albumentations import DualTransform
from albumentations.pytorch import transforms as T
from PIL import Image
from tools.image import (
    calculate_readout_noise,
    log_transform_scale,
    read_np_pil,
    standardize,
)
from tools.typing import *
from torch.utils.data import Dataset


class DivPadding(DualTransform):
    def __init__(self, always_apply=True, p=1.0, divisibility: int = 32):
        super().__init__(always_apply=always_apply, p=p)
        self.divisibility = divisibility

    def get_transform_init_args_names(self):
        return ["divisibility"]

    def get_params_dependent_on_targets(self, params):
        return params

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def pad(self, img: ndarray, pad_last: bool = False):
        dim = np.array(img.shape)
        if not pad_last:
            dim = dim[:-1]
        pad = np.ceil(dim / self.divisibility) * self.divisibility - dim
        pad = [(0, int(p)) for p in pad]
        if not pad_last:
            pad += [(0,) * len(dim)]
        # pad = np.stack([np.zeros_like(pad), pad[::-1]]).reshape(1, -1, order="F")[0]
        # pad = list(pad.astype(int))
        if np.allclose(pad, 0):
            return img
        return np.pad(img, pad, mode="reflect")

    def apply(self, img: ndarray, **params):
        return self.pad(img, pad_last=False)

    def apply_to_mask(self, img: ndarray, **params):
        return self.pad(img, pad_last=True)


class AnnotatedDataset(Dataset):
    def __init__(
        self,
        path: PathT = "../data/images/segmentation/",
        metainfo: PathT = "../data/segmentation_dataset_metainfo.csv",
        log: bool = True,
        std: bool = True,
        divisibility: int = 32,
        cache: bool = False,
        subtract_background_noise: bool = True,
        transforms: List[Transform] = [],
    ):
        self.path = Path(path)
        self.log = log
        self.std = std
        self.divisibility = divisibility
        self.cache = cache
        self.subtract_background_noise = subtract_background_noise
        self.original_shape = None

        fn_image = sorted(glob.glob(str(self.path / "input" / "Plate *" / "*.flex")))
        fn_label = sorted(
            glob.glob(str(self.path / "labels_dt" / "R2_Plate_*" / "*.npy"))
        )
        # print(len(fn_image), len(fn_label))
        genes = [f.split("/")[-1].split("_")[0] for f in fn_image]

        self.metainfo = pd.DataFrame(
            {
                "image": fn_image,
                "label": fn_label,
                "gene": genes,
            }
        )

        metainfo_gene = pd.read_csv(metainfo, names=["gene", "plate", "row", "column"])
        self.metainfo = self.metainfo.merge(metainfo_gene, on="gene", how="left")
        self.base_transforms = A.Compose(
            [
                DivPadding(divisibility=self.divisibility),
                T.ToTensorV2(),
            ]
        )
        self.aug_transforms = A.Compose(transforms)
        self.transforms = A.Compose([self.aug_transforms, self.base_transforms])

        # Background subtraction
        if self.subtract_background_noise:
            bg_fn = self.path / "background.npy"
            if bg_fn.exists():
                self.background = np.load(bg_fn)
            else:
                print("Background file does not exist, creating...")
                self.background = calculate_readout_noise(return_images=False)

    def __len__(self) -> int:
        return len(self.metainfo)

    def __getitem__(self, item: int) -> Tuple[Array, Array]:
        row = self.metainfo.iloc[item, :]
        image = read_np_pil(row["image"]).astype(np.float32)
        label = np.load(row["label"])

        self.original_shape = image.shape
        # print(image.shape, label.shape)
        if self.subtract_background_noise:
            image -= self.background
        if self.log:
            image = log_transform_scale(image)
        if self.std:
            image = standardize(image)

        transformed = self.transforms(image=image[..., None], mask=label)
        return transformed["image"].float(), transformed["mask"].float().unsqueeze(0)

    def eval(self):
        """
        Switch off training augmentations
        """
        self.transforms = A.Compose([self.base_transforms])

    def train(self):
        """
        Switch on training augmentations
        """
        self.transforms = A.Compose([self.aug_transforms, self.base_transforms])
