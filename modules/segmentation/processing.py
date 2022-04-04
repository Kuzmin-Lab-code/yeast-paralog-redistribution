import glob
import shutil
import sys

import numpy as np
from matplotlib.pyplot import imread
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from tqdm.auto import tqdm

from modules.tools.image import read_np_pil
from modules.tools.types import *


def semantic_to_binary(segmentation: Array) -> Array:
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


def make_distance_transform_dir(
    input_dir: PathT,
    output_dir: PathT,
    skip_existed: bool = True,
    alpha: float = 0.8,
    clip: int = 20,
    scale_by_stencil: bool = False,
) -> None:
    """
    Copies to
    :param input_dir: directory with object segmentation files
    :param output_dir: directory to save processed files
    :param skip_existed: if we skip existed files or rewrite them
    :param scale_by_stencil: bool, if to scale distance transform by stencil, scaled by image otherwise
    :param alpha: coefficient to combine stencils with distance transform
    :param clip: max value of distance transform before scaling
    :return: transformed segmentation map
    """
    assert output_dir != input_dir
    search_path = str(Path(input_dir) / "**/*.npy")
    filenames = sorted(glob.glob(search_path))
    iterator = tqdm(filenames)
    for fn in iterator:
        try:
            output_fn = Path(fn.replace(input_dir, output_dir))
            iterator.set_postfix({"input": str(fn), "output": str(output_fn)})
            if output_fn.exists() and skip_existed:
                continue
            output_fn.parents[0].mkdir(exist_ok=True, parents=True)
            s = np.load(fn)
            s = make_distance_transform(
                s, alpha=alpha, clip=clip, scale_by_stencil=scale_by_stencil
            )
            np.save(output_fn, s)
        except KeyboardInterrupt:
            break
        except:
            print(f"Unexpected error in {fn}:", sys.exc_info()[0])
            raise


def watershed_distance_map(
    distance_map: ndarray,
    seed_threshold: float = 0.8,
    mask_threshold: float = 0.5,
    region_assurance: bool = True,
    small_size_threshold: int = 32,
) -> ndarray:
    """
    Use watershed to binarize distance map
    :param distance_map: distance map either from algorithm or neural network output
    :param seed_threshold: threshold for watershed seeds
    :param mask_threshold: threshold for mask
    :param region_assurance: bool, preserves all the regions in mask
    :param small_size_threshold: size threshold to filter shall objects and holes
    :return: object segmentation
    """
    seed = remove_small_objects(distance_map > seed_threshold, small_size_threshold)
    mask = remove_small_holes(distance_map > mask_threshold, small_size_threshold)

    if region_assurance:
        uncertain_labels = label(mask)
        for lbl in np.unique(uncertain_labels)[1:]:
            stencil = np.array(uncertain_labels == lbl)
            if np.all(np.logical_not(np.logical_and(stencil, seed))):
                com = list(map(int, center_of_mass(stencil)))
                seed[com[0], com[1]] = True

    # Use minus, because watershed labels lowest values first
    ws = watershed(-distance_map, markers=label(seed), mask=mask, watershed_line=True)
    ws = remove_small_objects(ws, small_size_threshold)
    return ws


def extract_frames_from_image(
    image: ndarray, segmentation: ndarray, size: int = 64, mask_background: bool = False
) -> ndarray:
    """
    Crops square frames from image around stencil centroids in segmentation
    :param image: image with objects
    :param segmentation: binary segmentation
    :param size: size of frame
    :param mask_background: bool, make background zero
    :return: ndarray of frames
    """
    pad_image, pad_segmentation = map(
        lambda x: np.pad(x, size // 2), [image, segmentation]
    )
    if mask_background:
        pad_image[pad_segmentation < 0.5] = 0
    centroids = [p["centroid"] for p in regionprops(label(pad_segmentation))]

    n_obj = len(centroids)
    out = np.zeros((n_obj, size, size))

    for i, c in enumerate(centroids):
        s = tuple(slice(int(x - size // 2), int(x + size // 2)) for x in c)
        out[i] = pad_image[s]

    return out


def extract_frames_by_path(
    path_image: PathT = "../data/images/experiment/input/",
    path_segmentation: PathT = "../data/images/experiment/label/",
    path_frames: PathT = "../data/frames/",
    rewrite_existing: bool = False,
    **kwargs,
):
    """
    Extract frames from images using segmentation and preserving folder structure
    :param path_image: path to image root folder, search for .flex in subfolders
    :param path_segmentation: path to segmentation root folder, search for .png in subfolders
    :param path_frames: path to save extracted frames
    :param rewrite_existing: if False, skip existing files
    :param kwargs: for extract_frames_from_image()
    :return:
    """

    files_segmentation = sorted(glob.glob(f"{path_segmentation}/**/*."))
    for fs in tqdm(files_segmentation):
        fi = fs.replace(path_segmentation, path_image).replace("flex", "png")
        path_frames_from_image = Path(path_frames) / fs.split("/")[-1].split(".")[0]

        try:
            # Make path to frames, they will be stored in a folder with the name of parent file
            path_frames_from_image.mkdir(exist_ok=rewrite_existing, parents=True)

            # Read image and segmentation
            image = read_np_pil(fi)
            segmentation = imread(fs)

            # Extract frames from image
            frames = extract_frames_from_image(image, segmentation, **kwargs)

            # Save each frame in .npy file
            for i, frame in enumerate(frames):
                np.save(path_frames_from_image / f"{i:05d}", frame)

        except FileExistsError:
            # Skip existing directories
            continue

        except Exception as e:
            print(e)
            print(
                f"Error occurred in {path_frames_from_image}, remove the directory and break"
            )
            shutil.rmtree(path_frames_from_image)
            break
