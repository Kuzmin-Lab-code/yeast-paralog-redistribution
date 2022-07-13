import argparse
import glob
import json
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.segmentation import dataset, model
from modules.tools import util
from modules.tools.transforms import get_train_transforms
from modules.tools.util import SubsetSampler


def main():
    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument(
        "-r",
        "--run_path",
        help="Run path",
        type=str,
        default="results/segmentation/unet-resnet34/2022-07-12_22-48-28",
    )
    parser.add_argument(
        "--data_path", help="Test data path", default="data/images/experiment", type=str
    )
    parser.add_argument(
        "--validate", "-v", action="store_true", help="Run validation only"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        choices=["average", "last", "best"],
        default="average",
        help="Which model checkpoint to load, by default averages 3 best checkpoints and the last one",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        default=1,
        type=int,
    )
    parser.add_argument("--device", "-d", help="Device", default=0, type=int)
    parser.add_argument(
        "--postprocess",
        "-p",
        action="store_true",
        help="Postprocess masks with watershed",
    )

    args = parser.parse_args()

    # Paths
    cwd = Path(os.getcwd())
    print(f"Work in {args.run_path}")
    path = Path(args.run_path)
    target_path = path / "inference"
    target_path.mkdir(exist_ok=True)  # overwrite

    # Load config
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    if args.validate:
        print("Run validation")
        ads = dataset.AnnotatedDataset(
            path=cwd / cfg.data.path,
            metainfo=cwd / cfg.data.metainfo,
            log=cfg.data.log,
            std=cfg.data.std,
            distance_transform=cfg.data.distance_transform,
            transforms=get_train_transforms(cfg.data.crop),
        )
        ads.eval()
        # todo save indices during training
        indices = np.arange(len(ads))
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=cfg.training.valid_split,
            random_state=cfg.seed,
            shuffle=True,
            stratify=ads.metainfo.gene.tolist(),
        )
        loader = DataLoader(
            ads,
            batch_size=args.batch_size,
            num_workers=cfg.training.num_workers,
            sampler=SubsetSampler(valid_idx),
            pin_memory=True,
            drop_last=False,
        )
    else:
        # Assume experiment dataset
        ads = dataset.ExperimentDataset(
            path=args.data_path,
            log=cfg.data.log,
            std=cfg.data.std,
        )
        ads.eval()
        loader = DataLoader(
            ads,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=False,
        )

    mdl = model.SegmentationModel(
        **cfg.model,
        lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        epochs=cfg.training.epochs,
        classes_aux=ads.n_classes,
    )

    # Load checkpoints
    if args.checkpoint == "average":
        checkpoints = glob.glob(str(path / "**/**/checkpoints/*.ckpt"))
    elif args.checkpoint == "last":
        checkpoints = glob.glob(str(path / "**/**/checkpoints/last.ckpt"))
    elif args.checkpoint == "best":
        # todo not guaranteed to be best! double-check top-k saving pattern
        checkpoints = [glob.glob(str(path / "**/**/checkpoints/*.ckpt"))[-2]]
    else:
        raise ValueError(f"Checkpoint {args.checkpoint} is not supported")

    checkpoints = [torch.load(c)["state_dict"] for c in checkpoints]
    checkpoints = util.average_weights(checkpoints)
    mdl.load_state_dict(checkpoints)

    preds, ys = mdl.inference(loader)
    if args.validate:
        metrics = {
            k: f"{v.cpu().numpy(): .4f}" for k, v in mdl.valid_metrics.compute().items()
        }
        pprint(metrics)


if __name__ == "__main__":
    main()
