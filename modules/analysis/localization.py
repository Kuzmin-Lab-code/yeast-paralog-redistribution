import glob

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, kruskal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    log_loss,
    pairwise_distances,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tools.typing import List, Union
from tqdm.auto import tqdm, trange

Array = Union[np.ndarray]


class KnnClassifier:
    def __init__(
        self,
        X,
        y,
        k=5,
        metric="cosine",
        n_components=2,
        score_func=cohen_kappa_score,
        seed=56,
    ):
        self.metric = metric
        self.seed = seed
        self.score_func = score_func
        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.k = int(np.sqrt(np.min(self.classes_count) * 2)) if k is None else k

        self.tfm = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        self.X = self.tfm.fit_transform(X)

        self.distances = pairwise_distances(self.X, metric=self.metric)
        self.argsorted_distances = np.argsort(self.distances, axis=1)
        self.index = np.arange(len(self.distances))
        self.y = y
        self._y = y.copy()

    def _predict_idx(self, train_idx, test_idx):
        argsorted_distances = self.argsorted_distances[test_idx, :][:, train_idx]
        return self._predict(argsorted_distances)

    def predict_all(self):
        return self._predict(self.argsorted_distances)

    def _predict(self, argsorted_distances):
        predictions = self.y[argsorted_distances[:, : self.k]]
        predictions = np.apply_along_axis(
            np.bincount, 1, predictions, minlength=max(self.classes) + 1
        )
        predictions = np.argmax(predictions, 1)
        return predictions

    def train_score(self, *args, **kwargs):
        predictions = self.predict_all()
        return self.score_func(self.y, predictions, *args, **kwargs)

    def kfold_score(self, n_splits=5, *args, **kwargs):
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        scores = [
            self.score_func(
                self.y[test_idx],
                self._predict_idx(train_idx, test_idx),
                *args,
                **kwargs,
            )
            for train_idx, test_idx in kfold.split(self.y)
        ]
        return np.mean(scores)

    def permutation_test(self, n=1000, n_splits=None, *args, **kwargs):
        measurements = np.zeros(n + 1)
        for i in trange(n + 1):
            measurements[i] = (
                self.train_score(*args, **kwargs)
                if n_splits is None
                else self.kfold_score(n_splits, *args, **kwargs)
            )
            np.random.shuffle(self.y)
        self.y = self._y.copy()
        statistic = measurements[0]
        measurements = measurements[1:]
        return (
            {"statistic": statistic, "pvalue": np.mean(measurements >= statistic)},
            measurements,
        )


def aggregate(metainfo: pd.DataFrame, features: Array, group_by_replicate: bool = True):
    """
    Aggregate features by metainfo `label` column
    :param metainfo: metainfo about files
    :param features: extracted features
    :param group_by_replicate: extracted features
    :return:
    """
    assert (
        len(metainfo["label"].unique()) == 4
    ), "There should be 4 labels in metainfo for this analysis"

    features_df = pd.DataFrame(features)
    features_df["label"] = metainfo["label"]
    features_df["replicate"] = metainfo["replicate"]
    groups = ["replicate", "label"] if group_by_replicate else "label"

    return features_df.groupby(groups).mean()


def measure_pairwise_distance(
    aggregated_features: Array, pairs: List[str], metric: str = "cosine"
):
    # fmt: off
    """
    Measure distances between WT and DELTA conditions:
                         POR1-GFP POR2-WT 	POR2-GFP POR1-WT
    POR1-GFP POR2-DELTA 	0.047326 	        1.049365
    POR2-GFP POR1-DELTA 	0.811942        	1.097588
    :param aggregated_features: array of aggregated features (4, n_features)
    :param pairs: list of row names
    :param metric: metric to put in table
    :return: DataFrame, for both genes in pair measure distances
    """
    # fmt: on
    df = (
        pd.DataFrame(
            squareform(pdist(aggregated_features, metric=metric)),
            columns=pairs,
            index=pairs,
        )
        .drop([p for p in pairs if p.endswith("DELTA")], axis=1)
        .drop([p for p in pairs if p.endswith("WT")], axis=0)
    )
    return df


def stat_test_pc(features, labels, stat_test=kruskal, pc: int = 0, **kwargs):
    """Run univariate statistical test for the principal component"""
    tfm = PCA(n_components=1)
    tfm_features = tfm.fit_transform(features)
    feature = tfm_features[:, pc].squeeze()
    groups, group_counts = np.unique(labels, return_counts=True)

    # stat_test(delta, wt)
    res = {
        groups[0]
        .split("-")[0]: stat_test(
            feature[labels == groups[0]], feature[labels == groups[1]], **kwargs
        )
        ._asdict(),
        groups[2]
        .split("-")[0]: stat_test(
            feature[labels == groups[2]], feature[labels == groups[3]], **kwargs
        )
        ._asdict(),
    }
    res = pd.DataFrame(res).T
    res["test"] = stat_test.__name__

    return res, feature


def likelihood_ratio_test(features, labels, n_pca_components=None, **kwargs):
    """
    DF CALCULATED INCORRECTLY
    Fits two gaussian mixture models with 1 and 2 components respectively
    Returns p-value of statistical significance that 2-component model is better
    http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html
    """
    sample_weight = get_sample_weights(labels)

    # Reduce dimensionality (optional) for faster calculations
    tfm = PCA(n_components=n_pca_components)
    tfm_features = (
        tfm.fit_transform(features) if n_pca_components is not None else features
    )

    # Fit two gaussians with EM algorithm from initialized means
    means_init = [
        np.mean(tfm_features[labels == label], axis=0) for label in np.unique(labels)
    ]
    gm = GaussianMixture(n_components=2, means_init=means_init, **kwargs)
    gm.fit(tfm_features)

    # Null prob is majority voting, alternative is from the fitted model
    null_prob = np.round(sum(labels) / float(labels.shape[0]) * np.ones(labels.shape))
    alt_prob = gm.predict_proba(tfm_features)[:, 1]
    df = 1

    # Weighted log loss to mitigate class imbalance
    null_log_likelihood = -log_loss(
        labels, null_prob, normalize=True, sample_weight=sample_weight
    )
    alt_log_likelihood = -log_loss(
        labels, alt_prob, normalize=True, sample_weight=sample_weight
    )

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    return {"G": G, "pvalue": chi2.sf(G, df)}, alt_prob


# def test_pairs(pairs: List):
#     """
#     Test all the pairs in format KIN1-KIN2
#     :param pairs:
#     :return:
#     """
#     for pair in pairs:
#         g1, g2 = pair.split("-")
#         dataset = FramesDataset()


def get_sample_weights(labels):
    labels_unique, labels_counts = np.unique(labels, return_counts=True)
    labels_if = labels_counts[::-1] / labels_counts.sum()  # inverse frequency
    sample_weight = np.array([labels_if[i] for i in labels])
    sample_weight = sample_weight / np.sum(sample_weight) * len(sample_weight)
    return sample_weight


def likelihood_ratio_test_lr(
    features,
    labels,
    n_pca_components=None,
    normalize=True,
    weighted=False,
    **kwargs,
):
    """
    http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html
    """
    sample_weight = get_sample_weights(labels)

    # Reduce dimensionality (optional, depending on n_pca_components) for faster calculations
    tfm = make_pipeline(StandardScaler(), PCA(n_components=n_pca_components))
    tfm_features = tfm.fit_transform(features)

    # Define model
    lr_model = LogisticRegression(solver="lbfgs", max_iter=5000)
    lr_model.fit(tfm_features, labels)

    # fit intercept
    null_prob = np.mean(labels) * np.ones(labels.shape)
    alt_prob = lr_model.predict_proba(tfm_features)[:, 1]

    alt_log_likelihood = -log_loss(
        labels,
        alt_prob,
        normalize=normalize,
        sample_weight=sample_weight if weighted else None,
    )

    null_log_likelihood = -log_loss(
        labels,
        null_prob,
        normalize=normalize,
        sample_weight=sample_weight if weighted else None,
    )

    df = tfm_features.shape[1]  # degrees of freedom
    G = 2 * (alt_log_likelihood - null_log_likelihood)
    pvalue = chi2.sf(G, df)

    return (
        {
            "G": G,
            "pvalue": pvalue,
            "df": df,
            "log_likelihood_null": null_log_likelihood,
            "log_likelihood_altr": alt_log_likelihood,
        },
        alt_prob,
    )


def likelihood_ratio_test_gmm(
    features,
    labels,
    n_components=2,
    n_pca_components=None,
    normalize=True,
    weighted=False,
    n_init=3,
    **kwargs,
):
    """
    Fits two gaussian mixture models with 1 and 2 components respectively
    Returns p-value of statistical significance that 2-component model is better
    """
    sample_weight = get_sample_weights(labels)

    # Reduce dimensionality (optional, depending on n_pca_components) for faster calculations
    tfm = make_pipeline(StandardScaler(), PCA(n_components=n_pca_components))
    tfm_features = tfm.fit_transform(features)

    # Fit two gaussians with EM algorithm from initialized means
    means_init_altr = [
        np.mean(tfm_features[labels == label], 0) for label in np.unique(labels)
    ]
    means_init_null = [np.mean(tfm_features, 0)]

    gm_altr = GaussianMixture(
        n_components=n_components, means_init=means_init_altr, n_init=n_init, **kwargs
    )
    gm_null = GaussianMixture(
        n_components=1, means_init=means_init_null, n_init=n_init, **kwargs
    )

    gm_altr.fit(tfm_features)
    gm_null.fit(tfm_features)

    altr_prob = gm_altr.predict_proba(tfm_features)

    # Degrees of freedom of the model
    d = tfm_features.shape[1]
    df = (n_components - 1) * (d + d * (d - 1) / 2)

    # Calculate log likelihoods for two models
    log_likelihood_null = gm_null.score_samples(tfm_features)
    log_likelihood_altr = gm_altr.score_samples(tfm_features)

    if weighted:
        log_likelihood_null = np.multiply(log_likelihood_null, sample_weight)
        log_likelihood_altr = np.multiply(log_likelihood_altr, sample_weight)

    log_likelihood_null = np.mean(log_likelihood_null)
    log_likelihood_altr = np.mean(log_likelihood_altr)

    G = 2 * (log_likelihood_altr - log_likelihood_null)
    return (
        {
            "G": G,
            "pvalue": chi2.sf(G, df),
            "df": df,
            "log_likelihood_null": log_likelihood_null,
            "log_likelihood_altr": log_likelihood_altr,
        },
        altr_prob[:, 1],
    )


def collect_func_scores(
    func,
    score_name="score",
    path_pattern="../results/predictions/*-features.csv",
    **kwargs,
):
    features_files = sorted(glob.glob(path_pattern))
    results = []
    tfm = make_pipeline(StandardScaler(), PCA(n_components=2))

    for feature_file in tqdm(features_files[len(results) :]):
        features = pd.read_csv(feature_file, index_col=0)
        pair = feature_file.split("/")[-1].split("-")[:2]

        for i in range(2):
            wt = f"{pair[i]}-GFP {pair[1 - i]}-WT"
            dt = f"{pair[i]}-GFP {pair[1 - i]}-DELTA"

            features_sub = features.loc[
                (features.label == wt) | (features.label == dt), :
            ]
            label_sub = np.array(features_sub.label == wt, dtype=int)
            features_sub = features_sub.drop("label", axis=1)
            tfm_features = tfm.fit_transform(features_sub)

            score = func(tfm_features, label_sub, **kwargs)
            results.append(
                {
                    "pair": "-".join(pair),
                    "wt": wt,
                    "wt_count": np.sum(features.label == wt),
                    "delta": dt,
                    "delta_count": np.sum(features.label == dt),
                    score_name: score,
                }
            )

    results = pd.DataFrame(results)
    return results


def silhouette_test(features, labels, **kwargs):
    return silhouette_score(features, labels, **kwargs)


def knn_permutation_test(features, labels, **kwargs):
    knn = KnnClassifier(features, labels, **kwargs)
    res, measurements = knn.permutation_test()
    return res["pvalue"]


def sigclust_test(features, labels=None, **kwargs):
    try:
        import sigclust
    except ImportError:
        print("Install sigclust or provide path to it in sys")
        return
    return sigclust.sigclust(features, **kwargs)[0]


def collect_silhouette_scores(**kwargs):
    return collect_func_scores(silhouette_test, score_name="silhouette_score", **kwargs)


def collect_knn_permutation_pvalues(**kwargs):
    return collect_func_scores(knn_permutation_test, score_name="pvalue", **kwargs)


def collect_sigclust_pvalues(**kwargs):
    return collect_func_scores(sigclust_test, score_name="pvalue", **kwargs)
