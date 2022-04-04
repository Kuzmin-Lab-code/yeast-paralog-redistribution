import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

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
    fmt: str = "pdf",
) -> None:
    """
    Calculates protein abundance in all pairs
    :param update_metainfo: bool, to save updated metainfo with abundance scores
    :param reduce: reduce function (mean, median or max)
    :param plot: bool, to save boxplots
    :return:
    """
    metainfo = pd.read_csv(f"{str(meta_path)}/metainfo.csv")
    for pair in tqdm(np.unique(metainfo.pairs)):
        if pair.startswith("control"):
            # Ignore controls for now
            continue
        metainfo_pair = pd.read_csv(
            f"{str(meta_path)}/metainfo_replicate*_{pair}.csv", index_col=0
        )
        if "abundance" not in metainfo_pair.columns or force_update:
            abundance = calculate_intensity_list(metainfo_pair, reduce)
            metainfo_pair["abundance"] = abundance
            if update_metainfo:
                metainfo_pair.to_csv(f"{str(meta_path)}/metainfo_replicate*_{pair}.csv")
        if plot:
            plot_abundance_boxplots(
                metainfo_pair,
                save_path=save_path,
                separate_replicates=separate_replicates,
                save=True,
                fmt=fmt,
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
