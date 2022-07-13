import glob

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import transforms as T
from torch.utils.data import Dataset

from modules.tools.image import (
    calculate_readout_noise,
    log_transform_scale,
    read_np_pil,
    standardize,
)
from modules.tools.transforms import get_base_transforms
from modules.tools.types import *


class BaseDataset(Dataset):
    def __init__(
        self,
        path: PathT,
        log: bool = True,
        std: bool = True,
        cache: bool = False,
        divisibility: int = 32,
        subtract_background_noise: bool = True,
        transforms: Optional[List[Transform]] = None,
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
        self.base_transforms = get_base_transforms()
        self.transforms = self.base_transforms
        self.background = self._get_background()

        self.train()

    def _get_filenames(self) -> List[str]:
        """
        :return: list of images to iterate over
        """
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def _get_background(self) -> ndarray:
        if self.subtract_background_noise:
            bg_fn = Path(self.path) / "background.npy"
            if bg_fn.exists():
                self.background = np.load(bg_fn.as_posix())
            else:
                print("Background file does not exist, creating...")
                self.background = calculate_readout_noise(
                    self.files, return_images=False
                )
                np.save(bg_fn.as_posix(), self.background)
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
            raise ValueError("Original shape is not known, get some items first")
        slices = tuple(slice(s) for s in self.original_shape)
        # Pad dimensions
        # NB! Assuming torch channel first
        for _ in range(len(slices), len(x.shape)):
            slices = (slice(None),) + slices
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
        distance_transform: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(path=path, transforms=transforms, *args, **kwargs)
        print(f"Looking for data in {path}")

        fn_label = sorted(
            glob.glob(
                str(
                    self.path
                    / ("labels_dt" if distance_transform else "labels")
                    / "R2_Plate_*"
                    / "*.npy"
                )
            )
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

        self.metainfo.gene = pd.Categorical(self.metainfo.gene)
        self.metainfo["gene_id"] = self.metainfo.gene.cat.codes
        self.n_classes = self.metainfo["gene_id"].max() + 1

        metainfo_gene = pd.read_csv(metainfo, names=["gene", "plate", "row", "column"])
        self.metainfo = self.metainfo.merge(metainfo_gene, on="gene", how="left")

    def _get_filenames(self) -> List[str]:
        return sorted(glob.glob(str(self.path / "input" / "Plate *" / "*.flex")))

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        row = self.metainfo.iloc[i, :]
        mask = np.load(row["label"]).astype(np.float32)
        image = read_np_pil(row["image"]).astype(np.float32)
        image = self._normalize_image(image)

        item = self.transforms(image=image[..., None], mask=mask)
        item["mask"] = item["mask"].float().unsqueeze(0)
        item["label"] = torch.tensor(row["gene_id"]).long()
        return item


class ExperimentDataset(BaseDataset):
    def __init__(self, path: str = "../data/images/experiment", *args, **kwargs):
        super().__init__(path=path, *args, **kwargs)

    def _get_filenames(self) -> List[str]:
        return sorted(glob.glob(str(self.path / "**/*.flex")))

    def __getitem__(self, i) -> Dict[str, Tensor]:
        image = read_np_pil(self.files[i]).astype(np.float32)
        image = self._normalize_image(image)
        item = self.base_transforms(image=image[..., None])
        item["mask"] = torch.zeros_like(image)  # fill in label with 0 for compatibility
        item["label"] = -1
        return item
