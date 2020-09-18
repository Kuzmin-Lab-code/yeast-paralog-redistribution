import numpy as np
import pandas as pd
from tools.typing import *
from tools.viz import plot_abundance_boxplots
from tqdm.auto import tqdm


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
    update_metainfo: bool = True, reduce: str = "mean", plot: bool = True
) -> None:
    """
    Calculates protein abundance in all pairs
    :param update_metainfo: bool, to save updated metainfo with abundance scores
    :param reduce: reduce function (mean, median or max)
    :param plot: bool, to save boxplots
    :return:
    """
    metainfo = pd.read_csv("../data/metainfo.csv")
    for pair in tqdm(np.unique(metainfo.pairs)):
        if pair.startswith("control"):
            # Ignore controls for now
            continue
        metainfo_pair = pd.read_csv(
            f"../data/metainfo_replicate*_{pair}.csv", index_col=0
        )
        if "abundance" not in metainfo_pair.columns:
            abundance = calculate_intensity_list(metainfo_pair, reduce)
            metainfo_pair["abundance"] = abundance
            if update_metainfo:
                metainfo_pair.to_csv(f"../data/metainfo_replicate*_{pair}.csv")
        if plot:
            plot_abundance_boxplots(metainfo_pair, save=True)


def aggregate_protein_abundance(
    by: Union[str, Tuple[str]] = ("replicate", "label")
) -> DataFrame:
    """
    Aggregated protein abundances by mean of all scores per group
    :param by: list or str, defines group for aggregation
    :return: dataframe grouped by pair and provided arguments
    """
    metainfo = pd.read_csv("../data/metainfo.csv")
    if isinstance(by, str):
        by = tuple(
            by,
        )

    results = []
    for pair in tqdm(np.unique(metainfo.pairs)):
        if pair.startswith("control"):
            continue
        metainfo_pair = pd.read_csv(
            f"../data/metainfo_replicate*_{pair}.csv", index_col=0
        ).loc[:, ["pairs", *by, "abundance"]]
        metainfo_pair = metainfo_pair.groupby(["pairs", *by]).mean()
        results.append(metainfo_pair)

    results = pd.concat(results)
    return results


def relative_abundance_changes(aggregated_abundance: DataFrame) -> DataFrame:
    """
    Calculate relative abundance changes by gene pair and replicate
    :param aggregated_abundance: dataframe from `aggregate_protein_abundance`
    :return: dataframe with delta / wt intensity ratios
    """
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
