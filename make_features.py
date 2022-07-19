import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.classification import dataset, model
from modules.tools import util, viz


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument(
        "-r",
        "--run_path",
        help="Run path",
        type=str,
        default="results/classification/2022-07-18_15-50-57/",
    )
    parser.add_argument(
        "--metainfo_path",
        "-m",
        help="Metainfo path",
        default="data/meta/metainfo.csv",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        choices=["average", "last", "best"],
        default="average",
        help="Which model checkpoint to load, by default averages 3 best checkpoints and the last one",
    )
    parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument("--device", "-d", help="Device", default=0, type=int)

    args = parser.parse_args()

    # Paths
    cwd = Path(os.getcwd())
    print(f"Work in {args.run_path}")
    path = Path(args.run_path)

    target_path = path / "inference"
    features_path = target_path / "features"
    predictions_path = target_path / "predictions"
    metainfo_path = target_path / "metainfo"

    target_path.mkdir(exist_ok=args.overwrite)
    features_path.mkdir(exist_ok=args.overwrite)
    predictions_path.mkdir(exist_ok=args.overwrite)
    metainfo_path.mkdir(exist_ok=args.overwrite)

    # Load config
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # Read list of pairs from metainfo
    metainfo = pd.read_csv(args.metainfo_path, index_col=0)

    # Get model
    mdl = model.get_model(cfg, cwd=cwd, n_classes=metainfo.GFP.nunique())
    util.load_checkpoint_from_run_path(mdl, path, args.checkpoint)

    # Iterate over gene pairs
    iterator = tqdm(np.unique(metainfo.pairs))
    for pair in iterator:
        iterator.set_description(pair)
        select = pair.split("-")

        pair_dataset = dataset.FramesDataset(
            path_data=cfg.data.path_data,
            dir_frames=cfg.data.dir_frames,
            normalize=cfg.data.normalize,
            select=select,
        )

        pair_dataloader = DataLoader(
            pair_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )

        predictions, features, y = mdl.inference(pair_dataloader)

        fn = f"{pair}.csv"
        pd.DataFrame(features).assign(label=pair_dataset.metainfo.label).to_csv(
            features_path / fn,
        )
        pd.DataFrame(predictions).assign(label=pair_dataset.metainfo.label).to_csv(
            predictions_path / fn,
        )
        pair_dataset.metainfo.to_csv(
            metainfo_path / fn,
        )


if __name__ == "__main__":
    main()
