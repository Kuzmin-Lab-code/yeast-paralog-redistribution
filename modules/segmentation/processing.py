from types import Array

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label
from skimage.morphology import binary_erosion


def semantic_to_binary(segmentation: Array):
    """
    Transforms semantic segmentation to binary by eroding each object
    :param segmentation: semantic segmentation array, each object labeled with a distinct number
    :return: binary segmentation map with objects separated
    """
    return np.sum(
        [binary_erosion(segmentation == c) for c in np.unique(segmentation)[1:]],
        axis=0,
        dtype=bool,
    )


def make_three_class(segmentation: Array) -> Array:
    """
    Makes three-class segmentation from binary segmentation with binary erosion
    :param segmentation: segmentation image (N, M)
    :return: segmentation stack (N, M, 3) [background, foreground, border]
    """
    background = segmentation == 0
    if np.all(background):
        # Handle empty images
        return np.stack([background, ~background, ~background], axis=-1)
    # Erode each stencil to separate them
    foreground = semantic_to_binary(segmentation)
    border = (segmentation != 0) ^ foreground
    return np.stack([background, foreground, border], axis=-1)


def process_stencil(stencil: Array, distance_map: Array, alpha: float = 0.8) -> Array:
    """
    Combines a single binary stencil with distance transform by `alpha` coefficient
    stencil = stencil * alpha + distance_transform * (1-alpha)
    :param stencil: binary mask of a single cell
    :param distance_map: output of `distance_transform_edt()` (could be of a single cell or of a full image)
    :param alpha: coefficient to scale binary stencil and distance transform
    :return: weighted sum of binary stencil and its distance transform
    """
    assert 0 < alpha < 1, "Alpha should be a float between 0 and 1"
    assert (
        stencil.shape == distance_map.shape
    ), f"Arrays should have the same shape, not {stencil.shape} and {distance_map.shape}"
    stencil = binary_erosion(stencil)  # Erode stencil
    stencil_dt = distance_map.copy()  # Copy distance transform (dt) array
    stencil_dt[~stencil] = 0  # Leave only stencil area in dt
    stencil_dt -= stencil_dt.min()  # Scale dt between 0 and 1
    stencil_dt /= stencil_dt.max()
    return stencil * alpha + stencil_dt * (1 - alpha)


def make_distance_transform(
    segmentation: Array,
    alpha: float = 0.8,
    clip: int = 20,
    scale_by_stencil: bool = False,
) -> Array:
    """
    Makes distance transform from a segmentation array
    :param scale_by_stencil: bool, if to scale distance transform by stencil, scaled by image otherwise
    :param segmentation: semantic segmentation (N, M, 1) or three-class segmentation (N, M, 3)
    :param alpha: coefficient to combine stencils with distance transform
    :param clip: max value of distance transform before scaling
    :return: transformed segmentation map
    """
    if np.all((segmentation == 0)):
        return segmentation
    if len(segmentation.shape) == 2 or segmentation.shape[-1] == 1:
        # Process semantic segmentation
        # by eroding each stencil to separate them and summing binary maps
        segmentation_binary = semantic_to_binary(segmentation.squeeze())
    elif segmentation.shape[-1] == 3:
        # Take foreground class from three-class segmentation (already eroded)
        # segmentation_binary = binary_erosion(segmentation[..., 1].astype(np.float32))
        segmentation_binary = segmentation[..., 1].astype(np.float32)
    else:
        raise ValueError(
            f"Invalid segmentation shape {segmentation.shape} should have either 1 or 3 channels"
        )

    # Make distance transform, clip its values
    distance_map = distance_transform_edt(segmentation_binary)
    distance_map = np.clip(distance_map, 0, clip)

    # Sanity check of distance map
    ma = np.max(distance_map)
    if ma == 0:
        return segmentation_binary

    if scale_by_stencil:
        # Scale each stencil (takes time)
        segmentation_labeled, num_labels = label(segmentation_binary, return_num=True)
        out = np.sum(
            [
                process_stencil(segmentation == c, distance_map, alpha)
                for c in np.arange(1, num_labels)
            ],
            axis=0,
            dtype=np.float32,
        )
    else:
        # Scale distance map altogether
        distance_map -= np.min(distance_map)
        distance_map /= ma
        out = segmentation_binary * alpha + distance_map * (1 - alpha)
    return out
