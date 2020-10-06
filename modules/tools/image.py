import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .typing import *


def read_np_pil(path: PathT) -> ndarray:
    """
    Read an image to numpy array
    :param path: path to image
    :return: np array
    """
    return np.array(Image.open(path))


def min_max_scale(img: Array) -> Array:
    """
    Scales image between 0 and 1
    :param img: image to scale
    :return: scaled image
    """
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi + 1e-8)


def standardize(img: Array) -> Array:
    """
    Standardize image (subtract mean, divide by std)
    :param img: image to standardize
    :return: standardized image
    """
    return (img - img.mean()) / img.std()


def log_transform_scale(img: Array) -> Array:
    """
    Log transform image and scale it between 0 and 1
    :param img: image to transform
    :return:
    """
    tfm = np.log1p(img - img.min())
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
