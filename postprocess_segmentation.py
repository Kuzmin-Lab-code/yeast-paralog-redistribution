import argparse
import glob
from pathlib import Path

import numpy as np
from imageio import imwrite
from tqdm.auto import tqdm

from modules.segmentation.processing import watershed_distance_map


def main():
    parser = argparse.ArgumentParser(description="Segmentation postprocessing from probability maps")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        # TODO replace with default path
        default="results/segmentation/unet-resnet34/2022-07-14_00-20-42/inference",
        help="Path to segmentation",
    )
    parser.add_argument(
        "--postfix", type=str, default="_prob.npy", help="Postfix for glob search"
    )
    parser.add_argument(
        "--seed_threshold",
        type=float,
        default=0.8,
        help="Threshold to create watershed seeds",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Threshold to create watershed flooding area",
    )
    parser.add_argument(
        "--disable_region_assurance",
        action="store_true",
        help="By default, make sure that each region is segmented whether there is a seed inside or not",
    )
    parser.add_argument(
        "--small_size_threshold", type=int, default=256, help="Small size threshold"
    )
    parser.add_argument(
        "--large_size_threshold", type=int, default=8192, help="Large size threshold"
    )
    parser.add_argument(
        "--output_format", type=str, default="png", help="Output format"
    )
    args = parser.parse_args()

    fns = sorted(glob.glob(f"{args.path}/**/*{args.postfix}"))
    iterator = tqdm(fns)
    for fn in iterator:
        iterator.set_description(Path(fn).stem)
        prob = np.load(fn).squeeze()
        mask = watershed_distance_map(
            prob,
            seed_threshold=args.seed_threshold,
            mask_threshold=args.mask_threshold,
            region_assurance=not args.disable_region_assurance,
            small_size_threshold=args.small_size_threshold,
            large_size_threshold=args.large_size_threshold,
        )
        imwrite(fn.replace(args.postfix, f".{args.output_format}"), mask)


if __name__ == "__main__":
    main()
