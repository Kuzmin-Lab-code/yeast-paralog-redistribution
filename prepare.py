import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import trange, tqdm
import argparse
from typing import Tuple, List
from argparse import Namespace

import logging

def calculate_background(data_dir: str, replicates: Tuple[int, ...] = (1, 2, 3)) -> List[np.ndarray]:
    """
    Calculate background from data_dir
    """
    data_dir = Path(data_dir)
    bgs = []
    for i in tqdm(replicates):
        replicate = np.load(data_dir / f"replicate{i}.npy")
        print(replicate.shape)
        bg = np.median(replicate, axis=0)
        np.save(data_dir / f"background{i}.npy", bg)
        bgs.append(bg)
    return bgs


def extract_frames(data_dir: str, replicates: Tuple[int, ...] = (1, 2, 3)) -> None:
    """
    Extract frames from data_dir
    """
    data_dir = Path(data_dir)
    for i in tqdm(replicates):
        replicate = np.load(data_dir / f"replicate{i}.npy")
        print(replicate.shape)
        for j in range(replicate.shape[0]):
            np.save(data_dir / f"frames/frame{i}_{j}.npy", replicate[j])


def main(args: Namespace):
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(funcName)15s()] %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if "mode" in args:
        logger.info("Calculate background")
        calculate_background(args.data_dir)
    else:
        logger.info("Extract cell frames")
        extract_frames(args.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare data for training")
    parser.add_argument("--data_dir", "-d", type=str, default="data/images/experiment/input", )
    parser.add_argument("--replicates", "-r", type=list, nargs="+", default=[1, 2, 3],
                        help="Replicates to use for background calculation. Example: -r 1 2 3")

    subparsers = parser.add_subparsers()
    parser_bg = subparsers.add_parser(
        "background", help="Calculate background"
    )
    parser_frames = subparsers.add_parser(
        "frames", help="Extract cell frames"
    )

    args = parser.parse_args()
    print(args)
    main(args)

