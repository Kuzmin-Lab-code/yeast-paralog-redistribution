import argparse
import logging
from pathlib import Path

from modules.analysis.abundance import (
    calculate_mean_intensity_in_segmentation,
    normalize_abundance_percentile,
    normalize_abundance_percentile_segmentation,
    standardize_abundance,
)


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser("Normalize abundance")

    parser.add_argument(
        "--metainfo_path",
        "-p",
        help="Metainfo path, should have .csv files for each gene pair in question. NB! 'abundance' column is required",
        type=str,
        default="results/classification/2022-07-18_15-50-57/inference/metainfo",
    )

    subparsers = parser.add_subparsers(help="Normalization types", dest="normalization")
    parser_standardize = subparsers.add_parser(
        "std", help="Standardize abundance with mean and std"
    )

    parser_percentile = subparsers.add_parser(
        "percentile", help="Percentile-normalize abundance"
    )
    parser_percentile.add_argument(
        "--percentiles", nargs=2, default=[0.1, 99.9], help="Percentiles to use"
    )

    parser_segmentation = subparsers.add_parser(
        "segmentation", help="Percentile-normalize abundance with segmentation"
    )
    parser_segmentation.add_argument(
        "--percentiles", nargs=2, default=[0.1, 99.9], help="Percentiles to use"
    )
    parser_segmentation.add_argument(
        "--path_img",
        type=str,
        default="data/images/experiment/input",
        help="Path to raw images",
    )
    parser_segmentation.add_argument(
        "--path_seg",
        type=str,
        default="results/segmentation/unet-resnet34/2022-07-14_00-20-42/inference",
        help="Path to segmentation",
    )
    parser_segmentation.add_argument(
        "--segmentation_fmt", type=str, default="png", help="Segmentation format"
    )
    parser_segmentation.add_argument(
        "--image_fmt", type=str, default="flex", help="Segmentation format"
    )

    args = parser.parse_args()
    path = Path(args.metainfo_path)

    if args.normalization == "std":
        logger.info("Standardizing abundance")
        abundance_statistics = standardize_abundance(args.metainfo_path)
        abundance_statistics.to_csv(path.parent / "abundance_statistics_std.csv")

    elif args.normalization == "percentile":
        logger.info("Percentile-normalizing abundance")
        abundance_statistics = normalize_abundance_percentile(
            args.metainfo_path, *args.percentiles
        )
        abundance_statistics.to_csv(path.parent / "abundance_statistics_percentile.csv")

    elif args.normalization == "segmentation":
        logger.info("Percentile-normalizing abundance with segmentation")
        mean_intensity = calculate_mean_intensity_in_segmentation(
            path_img=args.path_img,
            path_seg=args.path_seg,
            image_fmt=args.image_fmt,
            segmentation_fmt=args.segmentation_fmt,
        )
        mean_intensity.to_csv(path.parent / "mean_intensity.csv")
        abundance_statistics = normalize_abundance_percentile_segmentation(
            args.metainfo_path, *args.percentiles, mean_intensity=mean_intensity
        )
        abundance_statistics.to_csv(
            path.parent / "abundance_statistics_percentile_segmentation.csv"
        )

    else:
        raise ValueError("Unknown normalization type")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(funcName)15s()] %(message)s",
        level=logging.INFO,
    )
    main()
