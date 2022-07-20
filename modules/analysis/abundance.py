import glob

import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from modules.tools.image import read_np_pil
from modules.tools.types import *
from modules.tools.viz import plot_abundance_boxplots


def calculate_intensity(img: Array, reduce: str = "mean") -> float:
    """
    Calculate intensity over frame (image) by reducing its values
    :param img: image
    :param reduce: reduce function (mean, median or max)
    :return:
    """
    assert reduce in ("mean", "median", "max")
    return eval(f"np.{reduce}")(img)


def calculate_intensity_list(files: List[str], reduce: str = "mean") -> List[float]:
    """
    Calculate intensity for a list of files
    :param files: list of paths to .npy files
    :param reduce: reduce function (mean, median or max)
    :return: list of intensities
    """
    return [calculate_intensity(np.load(fn), reduce) for fn in tqdm(files, leave=False)]


def calculate_protein_abundance(
    update_metainfo: bool = True,
    reduce: str = "mean",
    plot: bool = True,
    force_update: bool = False,
    separate_replicates: bool = False,
    meta_path: PathT = "./data/meta/",
    save_path: PathT = "./results/abundance/",
    abundance_col: str = "abundance",
    log_scale: bool = False,
    fmt: str = "pdf",
) -> None:
    """
    Calculates protein abundance in all pairs
    :param update_metainfo: bool, to save updated metainfo with abundance scores
    :param reduce: reduce function (mean, median or max)
    :param plot: bool, to save boxplots
    :param force_update: bool, to force update of metainfo with abundance column if it is already present
    :param separate_replicates: bool, to analyse replicates separately
    :param meta_path: path to metadata files for each pair
    :param save_path: path to save boxplots
    :param abundance_col: name of column to read/save abundance scores
    :param log_scale: bool, to use log scale for boxplots
    :param fmt: save format of boxplots (pdf, png, etc.)
    :return:
    """
    metainfo_files = sorted(glob.glob(f"{meta_path}/*.csv"))
    iterator = tqdm(metainfo_files, position=0)
    for fn in iterator:
        pair = Path(fn).stem
        iterator.set_description(pair)
        # Ignore controls for now
        if pair.startswith("control"):
            continue

        metainfo_pair = pd.read_csv(fn, index_col=0)
        if abundance_col not in metainfo_pair.columns or force_update:
            metainfo_pair[abundance_col] = calculate_intensity_list(
                metainfo_pair.file, reduce
            )
            if update_metainfo:
                metainfo_pair.to_csv(fn)

        if plot:
            plot_abundance_boxplots(
                metainfo_pair=metainfo_pair,
                save_path=save_path,
                separate_replicates=separate_replicates,
                save=True,
                fmt=fmt,
                abundance_col=abundance_col,
                log_scale=log_scale,
            )


def aggregate_protein_abundance(
    by: Union[str, Tuple[str]] = ("replicate", "label"),
    meta_path: PathT = "./data/meta/",
) -> DataFrame:
    """
    Aggregated protein abundances by mean of all scores per group
    :param by: list or str, defines group for aggregation
    :param meta_path: path to metadata
    :return: dataframe grouped by pair and provided arguments
    """
    metainfo = pd.read_csv(f"{meta_path}/metainfo.csv")
    if isinstance(by, str):
        by = tuple(
            by,
        )

    results = []
    for pair in tqdm(np.unique(metainfo.pairs)):
        if pair.startswith("control"):
            continue
        metainfo_pair = pd.read_csv(
            f"{meta_path}/metainfo_replicate*_{pair}.csv", index_col=0
        ).loc[:, ["pairs", *by, "abundance"]]
        metainfo_pair = metainfo_pair.groupby(["pairs", *by]).mean()
        results.append(metainfo_pair)

    results = pd.concat(results)
    return results


def relative_abundance_changes(
    aggregated_abundance: Optional[DataFrame] = None, *args, **kwargs
) -> DataFrame:
    """
    Calculate relative abundance changes by gene pair and replicate
    :param aggregated_abundance: dataframe from `aggregate_protein_abundance`
    :return: dataframe with delta / wt intensity ratios
    """
    if aggregated_abundance is None:
        aggregated_abundance = aggregate_protein_abundance(*args, **kwargs)

    results = []
    for (pair, replicate), df in tqdm(
        aggregated_abundance.reset_index().groupby(["pairs", "replicate"])
    ):
        df = df.reset_index()[["label", "abundance"]]
        df["GFP"] = df["label"].apply(lambda x: x.split("-")[0])
        for gene, group in df.groupby("GFP"):
            if len(group) == 2:  # Skip groups where data is unavailable
                # Calculate delta / wt intensity ratio
                abundance = group["abundance"].tolist()
                ratio = abundance[0] / abundance[1]
                results.append(
                    {"GFP": gene, "pair": pair, "replicate": replicate, "ratio": ratio}
                )

    results = pd.DataFrame(results)
    return results


def calculate_pca_abundance_correlation(
    pair: str,
    features_path: str = "./results/predictions-arc",
    meta_path: str = "./data/meta/",
    split_by_gene: bool = True,
) -> List[Dict[str, Any]]:
    """
    Calculate correlation between PCA and abundance scores for pair
    :param pair: pair to calculate correlation for, e.g. "KIN1-KIN2"
    :param features_path: path where features are stored in {pair}-features.csv files
    :param meta_path: path to metadata
    :param split_by_gene: split by gene in pair
    :return: list of dictionaries {pair, [gene], Pearson's r, p-value, Pearson's r absolute}
    """

    features_pair = pd.read_csv(
        f"{features_path}/{pair}-features.csv", index_col=0
    ).drop("label", axis=1)
    metainfo_pair = pd.read_csv(
        f"{meta_path}/metainfo_replicate*_{pair}.csv", index_col=0
    )

    results = []
    if split_by_gene:
        for gene in metainfo_pair.GFP.unique():
            mask = metainfo_pair.GFP == gene
            pc0 = PCA(n_components=1).fit_transform(features_pair[mask]).flatten()
            r, p = pearsonr(metainfo_pair.abundance[mask], pc0)
            results.append(
                {
                    "pair": pair,
                    "gene": gene,
                    "n": mask.sum(),
                    "r": r,
                    "p": p,
                    "rabs": np.abs(r),
                }
            )
    else:
        pc0 = PCA(n_components=1).fit_transform(features_pair).flatten()
        r, p = pearsonr(metainfo_pair.abundance, pc0)
        results.append({"pair": pair, "r": r, "n": len(pc0), "p": p, "rabs": np.abs(r)})
    return results


def calculate_pca_abundance_correlation_all_pairs(
    meta_path: PathT = "./data/meta/",
) -> DataFrame:
    """
    Calculate correlation between PCA and abundance scores for all pairs
    :param meta_path: path to metadata
    :return:
    """
    metainfo = pd.read_csv(
        f"{meta_path}/metainfo.csv", sep=",", index_col=0, dtype={"URL": object}
    ).reset_index()
    results = []
    for pair in tqdm(np.unique(metainfo.pairs)):
        if pair.startswith("control"):
            continue
        res = calculate_pca_abundance_correlation(pair)
        results.extend(res)
    return pd.DataFrame(results)


def calculate_mean_intensity_in_segmentation(
    path_img: Union[str, Path] = "data/images/experiment/input",
    path_seg: Union[
        str, Path
    ] = "results/segmentation/unet-resnet34/2022-07-14_00-20-42/inference",
    segmentation_fmt: str = "png",
    image_fmt: str = "flex",
) -> pd.DataFrame:
    """
    Mask background pixels with segmentation, calculate mean cell intensity in segmented areas
    :param path_img: path to images
    :param path_seg: path to segmentation
    :param segmentation_fmt: segmentation file format
    :param image_fmt: image file format
    :return:
    """
    mean_intensity = []
    files_image = sorted(glob.glob(f"{path_img}/**/*.{image_fmt}"))
    iterator = tqdm(files_image)
    for fi in iterator:
        fi = Path(fi)
        replicate = fi.parent.name
        name = fi.stem
        iterator.set_description(f"{replicate}/{name}")
        fs = Path(path_seg) / replicate / f"{name}.{segmentation_fmt}"

        if not fs.exists():
            raise FileNotFoundError(f"{fs} does not exist")

        image = read_np_pil(fi)
        segmentation = imread(fs)

        cell_pixels = image[segmentation > 0]
        mu = cell_pixels.mean() if len(cell_pixels) > 0 else 0
        n_pixels = len(cell_pixels)
        iterator.set_postfix(
            dict(img=image.shape, seg=segmentation.shape, n_pixels=n_pixels, mean=mu)
        )
        mean_intensity.append(dict(r=replicate, mu=mu, n_pixels=n_pixels))

    mean_intensity = pd.DataFrame(mean_intensity)
    mean_intensity["r"] = mean_intensity.r.astype("category")
    return mean_intensity


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = f"percentile_{n}"
    return percentile_


def normalize_abundance_percentile_segmentation(
    path_metainfo: Union[
        str, Path
    ] = "results/classification/2022-07-18_15-50-57/inference/metainfo",
    p_min: float = 0.1,
    p_max: float = 99.9,
    mean_intensity: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Normalize abundance scores to percentile of segmented cells
    abundance = (abundance - p_min) / (p_max - p_min)
    :param path_metainfo: path to metainfo .csv files for each pair
    :param p_min: low percentile
    :param p_max: high percentile
    :param mean_intensity: output of calculate_mean_intensity_in_segmentation()
    :param kwargs: passed to calculate_mean_intensity_in_segmentation() if mean_dict is None
    :return: abundance statistics (aggregated dataframe)
    """

    if mean_intensity is None:
        mean_intensity = calculate_mean_intensity_in_segmentation(**kwargs)

    abundance_statistics = (
        mean_intensity.loc[mean_intensity.n_pixels != 0, ["mu", "r"]]
        .groupby("r")
        .agg([percentile(p_min), percentile(p_max)])["mu"]
    )

    files_metainfo = sorted(glob.glob(f"{path_metainfo}/*.csv"))
    iterator = tqdm(files_metainfo)
    for fn in iterator:
        pair = Path(fn).stem
        iterator.set_description(pair)
        if pair == "control":
            continue

        metainfo_pair = pd.read_csv(fn, index_col=0)
        for replicate_id in range(1, 4):
            replicate = f"replicate{replicate_id}"
            p_min_value = abundance_statistics.loc[replicate, f"percentile_{p_min}"]
            p_max_value = abundance_statistics.loc[replicate, f"percentile_{p_max}"]

            abundance_repl_pnorm = (
                metainfo_pair.loc[metainfo_pair.replicate == replicate, "abundance"]
                - p_min_value
            ) / (p_max_value - p_min_value)

            metainfo_pair.loc[
                metainfo_pair.replicate == replicate, "abundance_repl_pnorm_seg"
            ] = abundance_repl_pnorm

        metainfo_pair.to_csv(fn)

    return abundance_statistics


def standardize_abundance(
    path_metainfo: Union[
        str, Path
    ] = "results/classification/2022-07-18_15-50-57/inference/metainfo",
) -> pd.DataFrame:
    """
    Standardize abundance scores with mean and std by replicate
    :param path_metainfo: path to metainfo .csv files for each pair
    :return: abundance statistics (aggregated dataframe)
    """

    files_metainfo = sorted(glob.glob(path_metainfo / f"*.csv"))
    iterator = tqdm(files_metainfo, desc="read abundance scores")

    abundances = [pd.read_csv(fn).loc[:, ["replicate", "abundance"]] for fn in iterator]
    abundances = pd.concat(abundances)
    abundance_statistics = abundances.groupby("replicate").agg(["mean", "std"])

    for fn in iterator:
        pair = Path(fn).stem
        iterator.set_description(pair)
        if pair == "control":
            continue

        metainfo_pair = pd.read_csv(fn, index_col=0)
        for replicate_id in range(1, 4):
            replicate = f"replicate{replicate_id}"
            mu = abundance_statistics.loc[replicate, "abundance"]["mean"]
            std = abundance_statistics.loc[replicate, "abundance"]["std"]

            abundance_repl_std = (
                metainfo_pair.loc[metainfo_pair.replicate == replicate, "abundance"]
                - mu
            ) / std
            metainfo_pair.loc[
                metainfo_pair.replicate == replicate, "abundance_repl_std"
            ] = abundance_repl_std

        metainfo_pair.to_csv(fn)

    return abundance_statistics


def normalize_abundance_percentile(
    path_metainfo: Union[
        str, Path
    ] = "results/classification/2022-07-18_15-50-57/inference/metainfo",
    p_min: float = 0.1,
    p_max: float = 99.9,
) -> pd.DataFrame:
    """
    Normalize abundance scores to percentile of all abundance scores
    :param path_metainfo: path to metainfo .csv files for each pair
    :param p_min: low percentile
    :param p_max: high percentile
    :return: abundance statistics (aggregated dataframe)
    """
    files_metainfo = sorted(glob.glob(path_metainfo / f"*.csv"))
    iterator = tqdm(files_metainfo, desc="read abundance scores")

    abundances = [pd.read_csv(fn).loc[:, ["replicate", "abundance"]] for fn in iterator]
    abundances = pd.concat(abundances)
    abundance_statistics = abundances.groupby("replicate").agg(
        [percentile(p_min), percentile(p_max)]
    )

    for fn in iterator:
        pair = Path(fn).stem
        iterator.set_description(pair)
        if pair == "control":
            continue

        metainfo_pair = pd.read_csv(fn, index_col=0)
        for replicate_id in range(1, 4):
            replicate = f"replicate{replicate_id}"
            p_min_value = abundance_statistics.loc[replicate, "abundance"][
                f"percentile_{p_min}"
            ]
            p_max_value = abundance_statistics.loc[replicate, "abundance"][
                f"percentile_{p_max}"
            ]

            abundance_repl_pnorm = (
                metainfo_pair.loc[metainfo_pair.replicate == replicate, "abundance"]
                - p_min_value
            ) / (p_max_value - p_min_value)
            metainfo_pair.loc[
                metainfo_pair.replicate == replicate, "abundance_repl_pnorm"
            ] = abundance_repl_pnorm

        metainfo_pair.to_csv(fn)

    return abundance_statistics
