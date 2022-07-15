import argparse
import pandas as pd

from modules.segmentation.processing import extract_frames_by_path


def main():
    parser = argparse.ArgumentParser(description="Frame extraction")
    parser.add_argument("--frame_size", "-f", type=int, default=64, help="Frame size")
    parser.add_argument("--mask", "-m", action="store_true", help="Mask background")
    parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite existing files"
    )

    parser.add_argument(
        "--segmentation",
        "-s",
        type=str,
        default="./data/segmentation/",
        help="Path to segmentation",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default="./data/images/experiment/input/",
        help="Path to images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/frames_updated/",
        help="Path to frames",
    )

    args = parser.parse_args()

    counts = extract_frames_by_path(
        path_segmentation=args.segmentation,
        path_image=args.images,
        path_frames=args.output,
        overwrite_existing=args.overwrite,
        size=args.frame_size,
        mask_background=args.mask,
    )

    counts = pd.DataFrame(counts)
    counts.to_csv(f"{args.output}/counts.csv")


if __name__ == "__main__":
    main()
