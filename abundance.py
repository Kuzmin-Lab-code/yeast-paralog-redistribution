import argparse
import logging
from pathlib import Path

from modules.analysis.abundance import calculate_protein_abundance


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        "Calculate abundance if necessary, plot boxplot comparison in gene pairs"
    )
    parser.add_argument(
        "--metainfo_path",
        "-p",
        help="Metainfo path, should have .csv files for each gene pair in question",
        type=str,
        default="results/classification/2022-07-18_15-50-57/inference/metainfo",
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["pairwise", "relative"],
        type=str,
        default="pairwise",
        help="Plot pairwise abundance changes in pairs or aggregated relative abundance changes",
    )

    parser.add_argument(
        "--reduce",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="how to reduce features and intensity values",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="force abundance recalculation even if column is present in pair metainfo",
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

    args = parser.parse_args()

    path = Path(args.metainfo_path)
    # Save nearby by default
    save_path = (
        path.parent / f"abundance-{args.format}-replicates={args.separate_replicates}"
    )
    logger.info(f"Save to {save_path}")

    calculate_protein_abundance(
        meta_path=path,
        reduce=args.reduce,
        force_update=args.force,
        plot=True,
        separate_replicates=args.separate_replicates,
        save_path=save_path,
        fmt=args.format,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(funcName)15s()] %(message)s",
        level=logging.INFO,
    )
    main()
