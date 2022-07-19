import argparse
import logging
from argparse import Namespace
from pathlib import Path

from modules.analysis.abundance import calculate_protein_abundance
from modules.tools.viz import plot_pca_all_pairs


def main(args: Namespace) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(funcName)15s()] %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if "mode" in args:
        logger.info("Abundance analysis")
        abundance(args)
    else:
        logger.info("Localization analysis")
        localization(args)


def abundance(args: Namespace) -> None:
    logger = logging.getLogger(__name__)
    if args.mode == "pairwise":
        logger.info("Run pairwise")
        save_path = f"{args.results_path}/abundance-{args.format}-replicates={args.separate_replicates}"
        logger.info(f"Save to {save_path}")
        calculate_protein_abundance(
            meta_path=args.meta_path,
            reduce=args.reduce,
            force_update=args.force,
            plot=True,
            separate_replicates=args.separate_replicates,
            save_path=save_path,
            fmt=args.format,
        )
        logger.info("Pairwise finished")
    elif args.mode == "relative":
        logger.info("Run relative")
        raise NotImplementedError

    else:
        logger.error("Wrong mode")
        raise ValueError("mode should be pairwise or relative")


def localization(args: Namespace) -> None:
    logger = logging.getLogger(__name__)
    save_path = f"{args.results_path}/localization-pca-{args.format}-replicates={args.separate_replicates}"
    logger.info(f"Save to {save_path}")
    plot_pca_all_pairs(
        metainfo_path=args.meta_path,
        features_path=args.features_path,
        fmt=args.format,
        separate_replicates=args.separate_replicates,
        save_path=save_path,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run localization and abundance analysis")

    # If we need to separate replicates in any analysis
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

    parser.add_argument("--meta_path", type=str, default="./data/meta")

    parser.add_argument("--results_path", type=str, default="./results")

    parser.add_argument(
        "--reduce",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="how to reduce features and intensity values",
    )

    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="overwrite existing files"
    )

    # make subparsers for localization and abundance
    subparsers = parser.add_subparsers()

    # analyze localization
    parser_loc = subparsers.add_parser(
        "localization", help="analyze protein localization"
    )

    parser_loc.add_argument(
        "--features_path",
        type=str,
        help="path to extracted features",
        default="./results/predictions-arc/",
    )

    # analyze abundance
    parser_ab = subparsers.add_parser("abundance", help="analyze protein abundance")

    parser_ab.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="force abundance recalculation even if column is present in pair metainfo",
    )
    parser_ab.add_argument(
        "--mode",
        "-m",
        choices=["pairwise", "relative"],
        type=str,
        required=True,
        help="plot pairwise abundance changes in pairs or aggregated relative abundance changes",
    )

    args = parser.parse_args()
    print(args)
    main(args)
