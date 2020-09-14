from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.models import HoverTool, Range1d, glyphs
from bokeh.plotting import (
    ColumnDataSource,
    figure,
    output_file,
    output_notebook,
    reset_output,
    show,
)
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def calculate_confidence_ellipse(x, y, sd: int = 2):
    params = dict(xy=(np.mean(x), np.mean(y)))
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    params["width"] = lambda_[0] * sd * 2
    params["height"] = lambda_[1] * sd * 2
    # params["angle"] = np.rad2deg(np.arccos(v[0, 0]))
    params["angle"] = np.degrees(np.arctan2(*v[:, 0][::-1]))

    return params


def draw_confidence_ellipse(ax, x, y, color: str = "black", sd: int = 2, lw: int = 2):
    ellipse_params = calculate_confidence_ellipse(x, y, sd)

    ell = Ellipse(
        xy=(ellipse_params["xy"]),
        width=ellipse_params["width"],
        height=ellipse_params["height"],
        angle=ellipse_params["angle"],
        facecolor="none",
        edgecolor=color,
        lw=lw,
        zorder=5,
    )

    cos_angle = np.cos(np.radians(180.0 - ellipse_params["angle"]))
    sin_angle = np.sin(np.radians(180.0 - ellipse_params["angle"]))

    xc = x - np.mean(x)
    yc = y - np.mean(y)

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (ellipse_params["width"] / 2.0) ** 2) + (
        yct ** 2 / (ellipse_params["height"] / 2.0) ** 2
    )

    ax.add_artist(ell)
    return ell, np.where(rad_cc <= 1)[0]


def get_confidence_ellipse_bokeh(x, y, color="black", sd=2, lw=2):
    ellipse_params = calculate_confidence_ellipse(x, y, sd)

    glyph = glyphs.Ellipse(
        x=ellipse_params["xy"][0],
        y=ellipse_params["xy"][1],
        width=ellipse_params["width"],
        height=ellipse_params["height"],
        angle=ellipse_params["angle"],
        fill_alpha=0,
        line_color=color,
        line_width=lw,
    )
    return glyph


def set_legend_marker_alpha(legend, alpha: float = 1):
    for lh in legend.legendHandles:
        lh._legmarker.set_alpha(alpha)


def plot_pca(
    features, metainfo, s: int = 7, use_seaborn=False, save_dir=None, scale=True
):

    # Define color scheme
    colors = np.array(["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a"])
    shapes = ["o", "^", "s"]

    # Transform features
    tfm = PCA(n_components=2)
    if scale:
        tfm = make_pipeline(StandardScaler(), tfm)
    features_tfm = tfm.fit_transform(features)

    df = pd.DataFrame(
        {
            "x": features_tfm[:, 0],
            "y": features_tfm[:, 1],
            "Label": metainfo.label,
            "Replicate": metainfo.replicate.str.replace("replicate", ""),
        }
    )

    label_order, label_count = np.unique(df.Label, return_counts=True)
    label_order = label_order[[1, 0, 3, 2]]  # WT comes first
    label_count = label_count[[1, 0, 3, 2]]  # WT comes first
    replicate_order, replicate_count = np.unique(df.Replicate, return_counts=True)
    fig, ax = plt.subplots(figsize=(s, s))

    # Default seaborn version
    if use_seaborn:
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="Label",
            style="Replicate",
            alpha=0.5,
            hue_order=label_order,
            s=15,
            ax=ax,
        )
    else:
        points_label = []
        for label, color in zip(label_order, colors):
            points_replicate_label = []
            for replicate, shape in zip(replicate_order, shapes):
                sub_df = df.loc[(df.Label == label) & (df.Replicate == replicate), :]
                # Scatter plot
                (points,) = ax.plot(
                    sub_df.x,
                    sub_df.y,
                    shape,
                    markersize=6,
                    label=label,
                    alpha=0.3,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )
                points_replicate_label.append(points)
            points_label.append(points_replicate_label.copy())

        # Confidence ellipse (for all replicates)
        for label, color in zip(label_order, colors):
            ellipse, within_ellipse = draw_confidence_ellipse(
                ax,
                df.loc[df.Label == label, "x"],
                df.loc[df.Label == label, "y"],
                color=color,
                lw=2,
            )

        # Legends
        # Replicates and shapes
        legend_replicate = ax.legend(
            points_label[-1],
            [f"Replicate {r} ({c})" for r, c in zip(replicate_order, replicate_count)],
            loc=2,
            markerscale=1.5,
            framealpha=0.8,
            fancybox=True,
            frameon=True,
        )
        legend_replicate.set_zorder(6)
        set_legend_marker_alpha(legend_replicate, 1)

        # Labels and colors
        legend_label = ax.legend(
            [lab[0] for lab in points_label if len(lab) != 0],
            [f"{r} ({c})" for r, c in zip(label_order, label_count)],
            loc=4,
            markerscale=1.5,
            framealpha=0.8,
            fancybox=True,
            frameon=True,
        )
        legend_label.set_zorder(6)
        set_legend_marker_alpha(legend_label, 1)

        ax.add_artist(legend_replicate)

    ax.set_xlabel("PCA 0")
    ax.set_ylabel("PCA 1")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pair = "-".join(np.unique(metainfo.GFP))
        plt.savefig(save_dir / f"{pair}.png", bbox_inches="tight")

    return fig, ax


def plot_abundance_boxplots(
    metainfo_pair,
    log_scale=True,
    separate_replicates=True,
    save=False,
    save_path="../results/abundance/",
):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharey=True)
    pair = np.unique(metainfo_pair.GFP)
    for i, (gene, ax) in enumerate(zip(pair, axes)):
        sns.boxplot(
            data=metainfo_pair[metainfo_pair.GFP == gene],
            x="label",
            y="abundance",
            hue="replicate" if separate_replicates else None,
            ax=ax,
        )
        labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.set_title(gene)
        if log_scale:
            ax.set(yscale="log")
    plt.tight_layout()
    if save:
        pair = "-".join(pair)
        plt.savefig(f"{save_path}/{pair}.png")
        fig.clear()
    else:
        plt.show()
