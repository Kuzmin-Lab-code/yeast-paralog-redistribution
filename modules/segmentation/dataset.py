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
import torch
from dataclasses import dataclass, field


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


class BaseDataset(Dataset):
    def __init__(
            self,
            path: PathT,
            log: bool = True,
            std: bool = True,
            cache: bool = False,
            divisibility: int = 32,
            subtract_background_noise: bool = True,
            transforms: Optional[List[Transform]] = None
    ):
        super().__init__()
        self.path = Path(path)
        self.log = log
        self.std = std
        self.cache = cache
        self.divisibility = divisibility
        self.subtract_background_noise = subtract_background_noise
        self.files = self._get_filenames()
        self.original_shape = None

        self.aug_transforms = transforms
        self.base_transforms = A.Compose(
            [
                DivPadding(divisibility=self.divisibility),
                T.ToTensorV2(),
            ]
        )
        self.transforms = self.base_transforms
        self.background = self._get_background()
        self.train()

    def _get_filenames(self) -> List[str]:
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def _get_background(self) -> ndarray:
        if self.subtract_background_noise:
            bg_fn = Path(self.path) / "background.npy"
            if bg_fn.exists():
                self.background = np.load(bg_fn)
            else:
                print("Background file does not exist, creating...")
                self.background = calculate_readout_noise(self.files, return_images=False)
                np.save(bg_fn, self.background)
        else:
            self.background = 0
        return self.background

    def _normalize_image(self, image):
        if self.original_shape is None:
            self.original_shape = image.shape
        if self.subtract_background_noise:
            image -= self.background
        if self.log:
            image = log_transform_scale(image)
        if self.std:
            image = standardize(image)
        return image

    def crop_to_original(self, x):
        if self.original_shape is None:
            raise ValueError('Original shape is not known, get some items first')
        slices = tuple(slice(s) for s in self.original_shape)
        # Pad dimensions
        # NB! Assuming torch channel first
        for _ in range(len(slices), len(x.shape)):
            slices = (slice(None), ) + slices
        return x[slices]

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


class AnnotatedDataset(BaseDataset):
    def __init__(
        self,
        path: PathT = "../data/images/segmentation/",
        metainfo: PathT = "../data/segmentation_dataset_metainfo.csv",
        transforms: Optional[List[Transform]] = None,
        *args,
        **kwargs
    ):
        super().__init__(path=path, transforms=transforms, *args, **kwargs)

        fn_label = sorted(
            glob.glob(str(self.path / "labels_dt" / "R2_Plate_*" / "*.npy"))
        )
        # print(len(fn_image), len(fn_label))
        genes = [f.split("/")[-1].split("_")[0] for f in self.files]

        self.metainfo = pd.DataFrame(
            {
                "image": self.files,
                "label": fn_label,
                "gene": genes,
            }
        )

        metainfo_gene = pd.read_csv(metainfo, names=["gene", "plate", "row", "column"])
        self.metainfo = self.metainfo.merge(metainfo_gene, on="gene", how="left")

    def _get_filenames(self) -> List[str]:
        return sorted(glob.glob(str(self.path / "input" / "Plate *" / "*.flex")))

    def __getitem__(self, item: int) -> Tuple[Array, Array]:
        row = self.metainfo.iloc[item, :]
        image = read_np_pil(row["image"]).astype(np.float32)
        label = np.load(row["label"])
        image = self._normalize_image(image)
        transformed = self.transforms(image=image[..., None], mask=label)
        return transformed["image"].float(), transformed["mask"].float().unsqueeze(0)


class ExperimentDataset(BaseDataset):
    def __init__(self, path: str = "../data/images/experiment", *args, **kwargs):
        super().__init__(path=path, *args,  **kwargs)

    def _get_filenames(self) -> List[str]:
        return sorted(glob.glob(str(self.path / '**/*.flex')))

    def __getitem__(self, item):
        image = read_np_pil(self.files[item]).astype(np.float32)
        image = self._normalize_image(image)
        image = self.base_transforms(image=image[..., None])['image']
        label = torch.zeros_like(image)  # fill in label with 0 for compatibility
        return image, label
