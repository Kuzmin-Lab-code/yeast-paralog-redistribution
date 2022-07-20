import argparse
import logging
from pathlib import Path

from modules.tools.viz import plot_pca_all_pairs


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        "Run localization analysis from extracted features"
    )
    parser.add_argument(
        "--inference_path",
        "-p",
        help="Inference path, should have folders 'features' and 'metainfo' with aligned files",
        type=str,
        default="results/classification/2022-07-18_15-50-57/",
    )

    parser.add_argument(
        "--separate_replicates",
        "-s",
        action="store_true",
        help="separate replicates in analysis",
    )

    parser.add_argument(
        "--format",
        "-t",
        type=str,
        choices=["pdf", "png"],
        default="pdf",
        help="format to store output",
    )

    parser.add_argument("--seaborn", action="store_true", help="Use seaborn")
    parser.add_argument("--figsize", type=int, default=7)

    args = parser.parse_args()

    path = Path(args.inference_path)
    # Save nearby by default
    save_path = (
        path / f"localization-pca-{args.format}-replicates={args.separate_replicates}"
    )
    logger.info(f"Save to {save_path}")

    plot_pca_all_pairs(
        metainfo_path=path / "metainfo",
        features_path=path / "features",
        fmt=args.format,
        separate_replicates=args.separate_replicates,
        save_path=save_path,
        use_seaborn=args.seaborn,
        figsize=args.figsize,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(funcName)15s()] %(message)s",
        level=logging.INFO,
    )
    main()
