import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from modules.tools.types import *


def read_np_pil(path: PathT) -> ndarray:
    """
    Read an image to numpy array
    :param path: path to image
    :return: np array
    """
    return np.array(Image.open(path))


def crop_as(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Crops x to shape evenly from each side
    (assumes even padding to remove)
    :param x:
    :param shape:
    :return: cropped x
    """
    n_dim = len(shape)
    bc_dim, img_dim = x.shape[:-n_dim], x.shape[-n_dim:]
    diff = np.array(img_dim) - np.array(shape)
    assert np.all(diff >= 0)
    top_left = diff // 2
    bottom_right = diff - top_left
    sl = tuple(slice(tl, s - br) for tl, s, br in zip(top_left, img_dim, bottom_right))
    sl = (slice(None), ) * len(bc_dim) + sl
    crop = x[sl]
    assert crop.shape[-n_dim:] == shape, f"Failed to crop to {shape}, output shape {crop.shape}"
    return crop


def min_max_scale(
    img: Array,
    mi: Optional[float] = None,
    ma: Optional[float] = None,
    eps: float = 1e-8,
) -> Array:
    """
    Scales image between 0 and 1
    :param img: image to scale
    :param mi: optional min, calculated as img.min() if None
    :param ma: optional max, calculated as img.max() if None
    :param eps: numerical stability in denominator
    :return: scaled image
    """
    if mi is None:
        mi = img.min()
    if ma is None:
        ma = img.max()
    return (img - mi) / (ma - mi + eps)


def standardize(
    img: Array, mean: Optional[float] = None, std: Optional[float] = None
) -> Array:
    """
    Standardize image (subtract mean, divide by std)
    :param img: image to standardize
    :param mean: optional mean to subtract, calculated as img.mean() if None
    :param std: optional std to divide, calculated as img.std() if None
    :return: standardized image
    """
    if mean is None:
        mean = img.mean()
    if std is None:
        std = img.std()
    return (img - mean) / std


def log_transform_scale(img: Array) -> Array:
    """
    Log transform image and scale it between 0 and 1
    :param img: image to transform
    :return:
    """
    tfm = img - img.min()
    if isinstance(img, ndarray):
        tfm = np.log1p(tfm)
    elif isinstance(img, Tensor):
        tfm = torch.log1p(tfm)
    return min_max_scale(tfm)


def calculate_readout_noise(
    filenames: List[str], return_images: bool = False
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    Calculates the read-out noise as a median across multiple images
    :param filenames: list of image filenames to be opened with Image.open
    :param return_images: whether to return all the images read as a second element
    :return: noise, (images - optional)
    """
    images = np.array([read_np_pil(fn) for fn in tqdm(filenames)])
    noise = np.median(images, axis=0)
    if return_images:
        return noise, images
    else:
        return noise
