import glob
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from skimage.color import label2rgb
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# from modules.analysis.abundance import relative_abundance_changes
from modules.tools.image import *
from modules.tools.metrics import *
from modules.tools.types import *


def clean_show(ax):
    """
    Shows ax without ticks (useful for images)
    :param ax: matplotlib axes
    :return:
    """
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def crop(
    img: Array,
    x: Optional[int] = None,
    y: Optional[int] = None,
    s: Optional[Union[int, Tuple[int]]] = None,
):
    """
    Crops a rectangle from full-size image
    :param img: image
    :param x: x coordinate of top-left corner
    :param y: y coordinate of top-left corner
    :param s: int or tuple, size of rectangle
    :return: cropped image
    """
    if crop is None:
        return img
    if isinstance(s, int):
        s = (s, s)
    if x is None:
        x = 0
    if y is None:
        y = 0
    assert x + s[0] < img.shape[0]
    assert y + s[1] < img.shape[1]
    return img[x : x + s[0], y : y + s[1]]


def plot_segmentation_overlay(
    img: Array,
    seg: Optional[Array] = None,
    crop_parameters: Optional[Tuple[int]] = None,
    size: int = 5,
    log_scale: bool = False,
):
    """
    Plots segmentation overlayed in RGB
    :param img: image
    :param seg: segmentation
    :param size: axes size
    :param crop_parameters: (x, y, s)
    :param log_scale: log transform (by default just min-max scale)
    :return:
    """
    # Consider image a segmentation
    if seg is None:
        seg = img
    nlabel = len(np.unique(seg))
    np.random.seed(30)
    colors = [tuple(map(tuple, np.random.rand(1, 3)))[0] for i in range(0, nlabel)]
    if crop_parameters is not None:
        img = crop(img.copy(), *crop_parameters)
        seg = crop(seg.copy(), *crop_parameters)
    ratio = img.shape[0] / img.shape[1]
    fig, ax = plt.subplots(figsize=(size * ratio, size))
    if log_scale:
        img = log_transform_scale(img)
    else:
        img = min_max_scale(img)
    labeled_img = label2rgb(seg, img, alpha=0.2, bg_label=0, colors=colors)
    ax.imshow(labeled_img)
    clean_show(ax)


def plot_by_side(
    images: List[Array],
    titles: Optional[List[str]] = None,
    crop_parameters: Optional[Tuple[int]] = None,
    size: int = 5,
    cmap="viridis",
):
    """
    Plot several images side by side
    :param images: list of images
    :param titles: list of titles, optional
    :param crop_parameters: (x, y, s), optional
    :param size: axes size
    :param cmap: pyplot colormap
    :return:
    """
    ratio = images[0].shape[0] / images[0].shape[1]
    ncols = len(images)
    fig, axes = plt.subplots(ncols=ncols, figsize=(size * ratio * ncols, size))
    for i, (img, ax) in enumerate(zip(images, axes)):
        if crop_parameters is not None:
            img = crop(img, *crop_parameters)
        ax.imshow(img, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i])
    clean_show(axes)


def calculate_confidence_ellipse(x: ndarray, y: ndarray, sd: int = 2):
    """
    Calculate confidence ellipse parameters: width, height, angle
    :param x: x coordinates
    :param y: y coordinates
    :param sd: standard deviations
    :return: dict of ellipse parameters
    """
    params = dict(xy=(np.mean(x), np.mean(y)))
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    params["width"] = lambda_[0] * sd * 2
    params["height"] = lambda_[1] * sd * 2
    # params["angle"] = np.rad2deg(np.arccos(v[0, 0]))
    params["angle"] = np.degrees(np.arctan2(*v[:, 0][::-1]))

    return params


def get_confidence_ellipse(
    x: ndarray, y: ndarray, color: str = "black", sd: int = 2, lw: int = 2
) -> Tuple[Ellipse, ndarray]:
    """
    Creates pyplot artist with confidence ellipse from x and y point coordinates
    Draw it with ax.add_artist(ellipse)
    :param x: x coordinates
    :param y: y coordinates
    :param color: line color
    :param sd: standard deviations
    :param lw: linewidth
    :return: Ellipse artist
    """
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

    return ell, np.where(rad_cc <= 1)[0]


def get_confidence_ellipse_bokeh(
    x: ndarray, y: ndarray, color: str = "black", sd: int = 2, lw: int = 2
) -> glyphs.Ellipse:
    """
    Creates bokeh glyph with confidence ellipse from x and y point coordinates
    :param x: x coordinates
    :param y: y coordinates
    :param color: line color
    :param sd: standard deviations
    :param lw: linewidth
    :return: bokeh glyph with confidence ellipse
    """
    from bokeh.models import glyphs
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
        lh.set_alpha(alpha)


def plot_pca(
    features: ndarray=None,
    metainfo: DataFrame=None,
    df: DataFrame=None,
    figsize: int = 7,
    use_seaborn: bool = False,
    save_path: Optional[str] = None,
    scale: bool = True,
    show_n:bool=True,
    separate_replicates: bool = False,
    include_replicates: Optional[Tuple[int]] = None,
    replicate_legend_loc: int = 2,
    label_legend_loc: int = 0,
    # Define color/shape scheme
    colors: np.array = np.array(["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a"]),
    shapes: list= ["o", "^", "s"],
    markersize: float=6,
    legend_markerscale: float=1.5,
    show_legends: bool= True,
    alpha: float=0.3,
    ax: plt.Axes=None,
    fmt: str = "pdf",
    kws_legend: dict={},
    **kws_scatter: dict,
):
    """
    Plot PCA of features array given metainfo data
    :param features: array of float features (N frames, N features)
    :param df: DataFrame containing the PCs, with the columns: 'x','y','Label' and 'Replicate'.
    :param metainfo: metainfo about each frame
    :param figsize: figure size
    :param markersize: size of the markers.
    :param alpha: transparency.
    :param use_seaborn: bool, produce basic plot with seaborn
    :param save_path: dir to save image, do not save if None
    :param scale: bool, apply standard scaling before PCA
    :param separate_replicates: bool, mark replicates with shapes
    :param include_replicates: tuple of int, indices of replicates to include
    :param label_legend_loc, location of label legend, in bottom-right corner by default (4)
    :param replicate_legend_loc, location of replicate legend, in top-left corner by default (2)
    :param fmt, format to save
    :param kws_legend, parameters provided to `plt.legend`.
    :param kws_scatter, keyword parameters provided to `sns.scatterplot` or `ax.plot`.
    :return:
    """
    
    if df is None:
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

    if include_replicates is not None:
        include_replicates = [str(i) for i in include_replicates]
        df = df[df.Replicate.isin(include_replicates)]

    label_order, label_count = np.unique(df.Label, return_counts=True)
    if len(label_order) > 3:
        label_order = label_order[[0, 1, 2, 3]]  #[[1, 0, 3, 2]]  # WT comes first
        label_count = label_count[[0, 1, 2, 3]]  #[[1, 0, 3, 2]]  # WT comes first
    replicate_order, replicate_count = np.unique(df.Replicate, return_counts=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Default seaborn version
    if use_seaborn:
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="Label",
            style="Replicate" if separate_replicates else None,
            s=markersize,
            alpha=alpha,
            hue_order=label_order,
            ax=ax,
            **kws_scatter,
        )
    else:
        points_label = []
        for label, color in zip(label_order, colors):
            points_replicate_label = []
            for replicate, shape in zip(replicate_order, shapes):
                if separate_replicates:
                    sub_df = df.loc[
                        (df.Label == label) & (df.Replicate == replicate), :
                    ]
                else:
                    sub_df = df.loc[df.Label == label, :]
                # Scatter plot
                (points,) = ax.plot(
                    sub_df.x,
                    sub_df.y,
                    shape,
                    markersize=markersize,
                    label=label,
                    alpha=alpha,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.1,
                    **kws_scatter,
                )
                points_replicate_label.append(points)
                if not separate_replicates:
                    break
            points_label.append(points_replicate_label.copy())

        # Confidence ellipse (for all replicates)
        for label, color in zip(label_order, colors):
            ellipse, within_ellipse = get_confidence_ellipse(
                df.loc[df.Label == label, "x"],
                df.loc[df.Label == label, "y"],
                color=color,
                lw=1,
            )
            ax.add_artist(ellipse)

        # Legends
        if show_legends:
            # Replicates and shapes
            if separate_replicates:
                legend_replicate = ax.legend(
                    points_label[-1],
                    [
                        f"Replicate {r} ({c})"
                        for r, c in zip(replicate_order, replicate_count)
                    ],
                    loc=replicate_legend_loc,
                    markerscale=legend_markerscale,
                    framealpha=0.8,
                    fancybox=True,
                    frameon=True,
                )
                legend_replicate.set_zorder(6)
                set_legend_marker_alpha(legend_replicate, 1)
                ax.add_artist(legend_replicate)

            # Labels and colors
            legend_label = ax.legend(
                [lab[0] for lab in points_label if len(lab) != 0],
                [f"{r}\n(n={c})" if show_n and nloc=='legend' else r.replace(' ','\n') for r, c in zip(label_order, label_count)],
                loc=label_legend_loc,
                markerscale=legend_markerscale,
                **{**dict(
                    frameon=False,
                    # framealpha=0.8,
                    # fancybox=True,
                    bbox_to_anchor=[1,1],
                    ),
                **kws_legend # overwite defaults with the inputs
                }
            )
            legend_label.set_zorder(6)
            set_legend_marker_alpha(legend_label, 1)
            ax.add_artist(legend_label)        
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        pair = "-".join(np.unique(metainfo.GFP))
        plt.tight_layout()
        # Save figure and point coordinates nearby
        plt.savefig(save_path / f"{pair}.{fmt}", bbox_inches="tight")
        df.to_csv(save_path / f"{pair}.csv")
    return ax


def plot_pca_all_pairs(
    metainfo_path: Union[str, Path] = "./data/meta/",
    features_path: Union[str, Path] = "./results/predictions-arc/",
    figsize: int = 7,
    use_seaborn: bool = False,
    save_path: Union[str, Path] = "./results/pca",
    scale: bool = True,
    separate_replicates: bool = False,
    replicate_legend_loc: int = 2,
    label_legend_loc: int = 4,
    fmt: str = "pdf",
):

    metainfo_files = sorted(glob.glob(f"{metainfo_path}/*.csv"))
    features_files = sorted(glob.glob(f"{features_path}/*.csv"))
    assert len(metainfo_files) == len(features_files), "Number of files mismatch"

    iterator = tqdm(zip(metainfo_files, features_files), total=len(metainfo_files))
    for mfn, ffn in iterator:
        pair = Path(ffn).stem
        iterator.set_description(pair)
        if pair.startswith("control"):
            continue

        metainfo_pair = pd.read_csv(mfn, index_col=0)
        features_pair = pd.read_csv(ffn, index_col=0).drop("label", axis=1)

        fig, ax = plot_pca(
            features=features_pair.values,
            metainfo=metainfo_pair,
            figsize=figsize,
            use_seaborn=use_seaborn,
            save_path=save_path,
            scale=scale,
            separate_replicates=separate_replicates,
            replicate_legend_loc=replicate_legend_loc,
            label_legend_loc=label_legend_loc,
            fmt=fmt,
        )
        plt.close(fig)


def plot_abundance_boxplots(
    metainfo_pair: DataFrame,
    log_scale: bool = False,
    separate_replicates: bool = True,
    save: bool = False,
    save_path: PathT = "../results/abundance/",
    abundance_col: str = "abundance",
    fmt: str = "pdf",
):
    """
    Plots abundance boxplots for gene pair side by side
    :param metainfo_pair: dataframe with abundance scores
    :param log_scale: scale y in log
    :param separate_replicates: separate boxplots for each replicate
    :param save: save figure
    :param save_path: where to save figure
    :param abundance_col: column name for abundance
    :param fmt: format to save figure
    :return:
    """
    assert (
        abundance_col in metainfo_pair.columns
    ), f"Abundance column {abundance_col} not found"
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharey=True)
    pair = np.unique(metainfo_pair.GFP)
    for i, (gene, ax) in enumerate(zip(pair, axes)):
        sns.boxplot(
            data=metainfo_pair[metainfo_pair.GFP == gene],
            x="label",
            y=abundance_col,
            hue="replicate" if separate_replicates else None,
            ax=ax,
        )
        labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.set_title(gene)
        if log_scale:
            ax.set(yscale="log")
            ax.set_ylim(bottom=1)
    plt.tight_layout()
    if save:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        pair = "-".join(pair)
        name = f"{save_path}/{pair}.{fmt}"
        plt.tight_layout()
        plt.savefig(name)
        # fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_relative_changes(
    relative_changes: Optional[DataFrame] = None,
    offset_thr: float = 1.0,
    *args,
    **kwargs,
):
    """
    Plot relative_abundance_changes() output as a boxplot
    :param relative_changes:
    :param offset_thr: threshold to highlight
    :return:
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    # if relative_changes is None:
    #     relative_changes = relative_abundance_changes(*args, **kwargs)
    relative_changes = relative_changes.dropna()
    coords = np.log(relative_changes["ratio"])
    sns.stripplot(coords, linewidth=1, ax=ax, jitter=0.2)
    sns.boxplot(coords, fliersize=0, whis=3, boxprops=dict(alpha=0.2))

    interesting = []
    for c in ax.collections:
        for name, replicate, offset in zip(
            relative_changes.GFP.tolist(),
            relative_changes.replicate.tolist(),
            c.get_offsets(),
        ):
            if offset[0] < -offset_thr or offset[0] > offset_thr:
                name = f"{name} (R{replicate[-1]})"
                ax.annotate(name, offset)
                interesting.append(name)

    plt.tight_layout()
    ax.set_xlabel("log relative intensity (delta / WT)")
    plt.show()

def plot_umap(
    df3:pd.DataFrame,
    show_outlines: bool=False,
    )->plt.Axes:
    """
    Plot UMAP.
    """
    from roux.viz.colors import saturate_color
    palette=dict(zip(sorted(df3['construct label'].unique()),
    [
    saturate_color(metadata['colors']['gene1'],0.5),
    metadata['colors']['gene1'],
    saturate_color(metadata['colors']['gene2'],0.5),
    metadata['colors']['gene2'],
    ]))
    fig,ax=plt.subplots(figsize=[3,3])
    ax=sns.scatterplot(
        data=df3,
        x='x',
        y='y',
        hue='construct label',
        ec=None,
        alpha=0.5,
        s=5,
        hue_order=palette.keys(),
        palette=palette.values(),
        )
    if show_outlines:
        sns.kdeplot(
            data=df3, 
            x="x", 
            y="y", 
            hue="construct label",
            ax=ax,
            alpha=0.2,
            levels=[0.25],
            hue_order=palette.keys(),
            palette=palette.values(),
            )
    ax.legend(bbox_to_anchor=[1,1],frameon=False)
    ax.grid(False)
    ax.axis('off')
    ax.margins(tight=True)
    off_arrow=0.2#.25
    ax.arrow(x=off_arrow, y=off_arrow, dx=0.1, dy=0, head_width = .02,transform=fig.transFigure,clip_on=False,color='k',lw=1)
    ax.arrow(x=off_arrow, y=off_arrow, dx=0, dy=0.1, head_width = .02,transform=fig.transFigure,clip_on=False,color='k',lw=1)
    ax.text(x=off_arrow,y=off_arrow-0.01,s="UMAP1",transform=fig.transFigure,ha='left',va='top')
    ax.text(y=off_arrow,x=off_arrow-0.01,s="UMAP2",transform=fig.transFigure,rotation=90,ha='right',va='bottom')
    return ax
