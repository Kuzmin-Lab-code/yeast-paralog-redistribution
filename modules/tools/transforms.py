import albumentations as A
import numpy as np
from albumentations import DualTransform
from albumentations.pytorch import transforms as T
from numpy import ndarray


def get_train_transforms(crop: int = 256):
    transforms = A.Compose(
        [
            A.RandomCrop(crop, crop),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
        ]
    )
    return transforms


def get_base_transforms(divisibility: int = 32):
    transforms = A.Compose(
        [
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=divisibility,
                pad_width_divisor=divisibility,
            ),
            # DivPadding(divisibility=self.divisibility),
            T.ToTensorV2(),
        ]
    )
    return transforms


class DivPadding(DualTransform):
    """
    Pads image and mask to be divisible by some value
    NB! PadIfNeeded has the same functionality but only in dev version,
    so this class will be deprecated after the next Albumentations release
    """

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
