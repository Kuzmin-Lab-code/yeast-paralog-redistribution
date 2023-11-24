#!/usr/bin/env python
"""
Plotting functions.
"""
import logging
## sys
from os.path import splitext, dirname, exists, basename
from roux.lib.sys import to_path, basenamenoext
## data
import itertools
import numpy as np
import pandas as pd
from roux.lib.str import replace_many
from roux.lib.dict import flip_dict
from roux.lib.set import list2str, get_pairs
## plotting
import seaborn as sns
import matplotlib.pyplot as plt
from roux.viz.ax_ import get_axlims,get_ticklabel_position,set_label
# i/o
from roux.lib.io import read_table, read_dict, to_table
from roux.viz.io import to_plot
## local functions
from .viz import plot_umap

def annotate_scalebar(
    ax: plt.Axes,
    pixels: float, # in pixels
    label: str,
    prefix: str='scale bar = ',
    img=None,
    xlim:list =None,
    ylim:list =None,
    x: float=None,
    y: float=None,
    color='w',
    off_scale=0.05, # scale bar is off_scale fraction away from the axes
    ) -> plt.Axes:
    """Annotate the scale bar on the images

    Args:
        ax (plt.Axes): subplot object.
        pixels (float): pixel width.
        prefix (str, optional): prefix of the label. Defaults to 'scale bar = '.
        img (_type_, optional): image. Defaults to None.
        xlim (list, optional): x-axis limits. Defaults to None.
        ylim (list, optional): y-axis limits. Defaults to None.
        x (float, optional): x position. Defaults to None.
        y (float, optional): y position. Defaults to None.
        color (str, optional): color. Defaults to 'w'.
        off_scale (float, optional): offset scale. Defaults to 0.05.

    Returns:
        plt.Axes: subplot
    """
    if x is None and y is None:
        if not img is None:
            ## offset from the bottom right corner 
            x=img.shape[1]-np.min(np.ceil(np.array(img.shape)*(off_scale*0.5)))
            y=img.shape[0]-np.min(np.ceil(np.array(img.shape)*off_scale))
        elif not (xlim is None and ylim is None):
            fliph=xlim!=sorted(xlim)
            flipv=ylim!=sorted(ylim)                
            xlim,ylim=sorted(xlim),sorted(ylim)
            if not fliph:
                x=xlim[1]-int((xlim[1]-xlim[0])*(off_scale*0.5))
            else:
                x=xlim[0]+int((xlim[1]-xlim[0])*(off_scale*0.5))
            if not flipv:
                y=ylim[1]-int((ylim[1]-ylim[0])*(1-off_scale))
            else:
                y=ylim[0]+int((ylim[1]-ylim[0])*(1-off_scale))                
    logging.info(xlim,ylim,x,y)
    # if test: 
    xs=[x,x+((-1 if not fliph else 1)*pixels)]
    ax.plot(
        xs,
        [y,y],
        color=color,
        solid_capstyle='butt',
        )
    ax.text(
        s=prefix+label,
        transform=ax.transAxes,
        x=1,
        y=-0.01,
        ha='right',
        va='top',
        
        # x=np.mean(xs),
        # y=y,
        # ha='center',
        # va='top',
        # color='',
        clip_on=False,
        )
    return ax

def plot_image_file(
    image_id: str,
    path_raw_images: str="data/images/experiment/input/replicate3/",
    path_segmented_images: str='data/images/experiment/label/replicate3/',
    ext_raw_images: str='flex',
    ext_segmented_images: str='png',
    cmap: str='gfp',
    cutoff_vmax: float=None, # quantile cut off
    cutoff_vmax_data = None, 
    color_segmentation_outline: str='w',
    linewidths: float=1,
    show_segmentation: bool=True,
    test: bool=False,
    ax: plt.Axes=None,
    **kws
    ) -> plt.Axes:
    """
    Plot the image.
    
    Args:
        image_id (str): image ID.
        path_raw_images (str): path to the raw iamges.
        path_segmented_images (str): path to the segmented images.
        ext_raw_images (str): extension of the raw images. 
        ext_segmented_images (str): extension of the segmented images.
        cmap (str): colormap.
        cutoff_vmax (float): cutoff to set the maximnum intensity. 
        cutoff_vmax_data (np.array): data for calculating cutoff to set the maximnum intensity, else based on the current image.
        color_segmentation_outline (str): color of the segmentation outline.
        linewidths (float): width of the lines.
        show_segmentation (bool): show the segmentation lines.
        test (bool): test mode.
        ax (plt.Axes): subplot object.
    
    Keyword Args:
        kws (dict): keyword arguments provided to the `image_background` function.
    
    Returns:
        subplot object.
    """
    from htsimaging.viz.image import image_background
    
    from skimage.io import imread
    if show_segmentation:
        segmented_image_path=f"{path_segmented_images}/{image_id}.{ext_segmented_images}"
        if not exists(segmented_image_path):
            logging.error(f"not found {segmented_image_path}")
            show_segmentation=False
        a1=imread(segmented_image_path)
    # gfp image
    path_raw_image=f"{path_raw_images}/{image_id}.{ext_raw_images}"
    # if not exists(path_raw_image):
    #     logging.warning(f'image not found: {path_raw_image}')
    #     return
    # print(f"imread('{path_raw_image}',as_gray=True)")
    a2=imread(
        path_raw_image,
        as_gray=False,
        )#[0]
    if test:
        logging.info(a2.shape,a2.max())
        plt.imshow(a2)
        plt.imshow(a1)    
    if show_segmentation:
        assert a1.shape==a2.shape, (a1.shape,a2.shape)
        from skimage import measure
        regions=measure.label(a1)
    else:
        regions=None
    if test:
        logging.info(a1.shape)
        ## get regions i.e. cell boundaries
        # from htsimaging.lib.segment import segmentation2cells
        # regions=segmentation2cells(
        #     imp=a2,
        #     imsegp=a1,
        #     fiterby_border_thickness=100,
        #     magnification=100,
        #     # plotp='test/segmentation2cells',
        # )
        # from skimage import io
        # io.imread(imsegp,as_gray=True)     
    if ax is None:
        fig,ax=plt.subplots(figsize=[8,5])
    # from htsimaging.lib.plot import image_background
    if test:
        logging.info(cutoff_vmax_data,cutoff_vmax)
    if not cutoff_vmax is None:
        # quantile cut off to intensity cutoff
        kws['vmax']=getattr(np,'quantile')(a2 if cutoff_vmax_data is None else cutoff_vmax_data,cutoff_vmax)            
        if test:
            logging.info(kws['vmax'])
    ax=image_background(
        img_region=regions,
        img=a2,
        cmap=cmap,
        linewidths=linewidths,
        colors=color_segmentation_outline,
        alpha=1,
        show_cbar=False,
        ax=ax,
        **kws,
    )
    # ax.axis('off')
    ax.set(
        xlabel=None,ylabel=None,
        xticks=[],yticks=[],
        xticklabels=[],yticklabels=[],
    )
    return ax

def plot_image_cropped(
    URL: str,#='009013004',
    replicate,#='replicate1',
    position: list,#=[365, 446],
    size: int,
    flipv: bool=False,
    fliph: bool=False,
    # rotation: float=None,
    path_raw_images: list=None,
    path_segmented_images: list=None,
    ext_raw_images: str='flex',
    show_scalebar: bool=None,
    ax: plt.Axes=None,
    kws_annotate_scalebar:dict={},
    test: bool=False,    
    **kws: dict,
    ) -> plt.Axes:
    """Plot cropped image.

    Args:
        URL (str): image ID.
        replicate (str): replicate name.
        position (list): x and y co-ordinates of the top left corner.
        size (int): size of the image.
        flipv (bool, optional): flip vertically. Defaults to False.
        fliph (bool, optional): flip horizontally. Defaults to False.
        rotation (float, optional): rotation. Defaults to None.
        path_raw_images (list, optional): path to the raw images. Defaults to None.
        path_segmented_images (list, optional): path to the segmented images. Defaults to None.
        ext_raw_images (str, optional): extension of the raw images. Defaults to 'flex'.
        show_scalebar (bool, optional): show scalebar. Defaults to None.
        ax (plt.Axes, optional): subplot. Defaults to None.
        kws_annotate_scalebar (dict, optional): keyword arguments provided to scalebar annotation. Defaults to {}.
        test (bool, optional): test mode. Defaults to False.

    Keyword Args:
        keyword arguments provided to the `plot_image_file` function.
    
    Returns:
        plt.Axes: subplot
    """
    ax=plot_image_file(
        image_id=URL,
        path_raw_images=f"{path_raw_images}/{replicate}/",
        path_segmented_images=f"{path_segmented_images}/{replicate}/",
        ext_raw_images=ext_raw_images,
        ext_segmented_images='png',
        test=test,
        ax=ax,
        **kws
        )   
    xlim=[position[0],position[0]+size]
    if fliph:
        xlim=xlim[::-1]
    ylim=[position[1]+size,position[1]]
    if flipv:
        ylim=ylim[::-1]        
    ax.set(
        xlim=xlim,
        ylim=ylim,
          )            
    if not show_scalebar is None:
        ax=annotate_scalebar(
            ax,
            pixels=list(show_scalebar.values())[0],
            label=list(show_scalebar.keys())[0],
            xlim=xlim,ylim=ylim,
            **kws_annotate_scalebar,
            )    
    return ax

def arrange_images_cropped(
    pair: str,
    path_raw_images: str,
    ext_raw_images: str,
    config_cropping: dict,
    axd: dict=None,
    fig: plt.Figure=None,
    panel: str=None,
    # image_type: str='raw',
    image_scale: float=None,
    cutoff_vmax_data: bool=False, # infer cutoff from the data
    cutoff_vmax_by_query: bool=True, # calculate cutoff from the query
    # show_construct_label_colors=True,
    labels_loc: str='top',
    queryprefix2color: dict={},
    show_wt: bool=False,
    ## kws_plot_image_cropped
    test: bool=False,
    **kws_plot_image_cropped,
    ) -> dict:
    """Arrange the cropped images.

    Args:
        pair (str): name of the gene pair.
        path_raw_images (str): path to the raw images.
        ext_raw_images (str): extensions of the raw images.
        config_cropping (dict): configuration for cropping of images.
        axd (dict, optional): subplots. Defaults to None.
        fig (plt.Figure, optional): figure. Defaults to None.
        panel (str, optional): panel name. Defaults to None.
        image_scale (float, optional): scaling of the images. Defaults to None.
        cutoff_vmax_data (bool, optional): cutoff on the max intensity based on data. Defaults to False.
        queryprefix2color (dict, optional): query to color mapping. Defaults to {}.
        show_wt (bool, optional): show wild-type. Defaults to False.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        dict: subplots
    """
    if axd is None:
        panel='A',
        if fig is None:
            fig=plt.figure(figsize=[4,4])
        axd={
            f'{panel}':plt.subplot2grid(shape=[2,2], loc=[0,0], rowspan=1, colspan=1,fig=fig),
            f'{panel}2':plt.subplot2grid(shape=[2,2], loc=[0,1], rowspan=1, colspan=1,fig=fig),
            f'{panel}3':plt.subplot2grid(shape=[2,2], loc=[1,0], rowspan=1, colspan=1,fig=fig),
            f'{panel}4':plt.subplot2grid(shape=[2,2], loc=[1,1], rowspan=1, colspan=1,fig=fig),
            }        
    def construct_label2query(k): return (k.split('p-GFP')[0]).upper()
    if cutoff_vmax_data:
        ### get all the images to calculate the cutoff_vmax from 
        ## get the list of paths
        # ps={k:f"{metadata['data']['images'][image_type]['path']}/{d['replicate']}/{d['URL']}.{metadata['data']['images'][image_type]['ext']}" for k,d in config_cropping[pair].items()}
        ps={k:f"{path_raw_images}/{d['replicate']}/{d['URL']}.{ext_raw_images}" for k,d in config_cropping[pair].items()}
        ## read arrays
        from skimage.io import imread
        ims={k:imread(p,as_gray=False,) for k,p in ps.items()}
        if cutoff_vmax_by_query:
            logging.info({k:construct_label2query(k) for k in ims})
            query2constructs=flip_dict({k:construct_label2query(k) for k in ims})
            logging.info(query2constructs)
            cutoff_vmax_data={}
            for q,cs in query2constructs.items():
                cutoff_vmax_data[q] = np.concatenate([ims[c] for c in cs])
        else:
            ## concat the arrays
            cutoff_vmax_data=np.concatenate(ims)
    else:
        cutoff_vmax_data=None
        
    for i,(construct_label,ax) in  enumerate(zip(list(config_cropping[pair].keys()),[axd[f'{panel}'],axd[f'{panel}2'],axd[f'{panel}3'],axd[f'{panel}4']])):
        ax=plot_image_cropped(
            path_raw_images=path_raw_images,
            ext_raw_images=ext_raw_images,            
            show_scalebar=image_scale if i==3 else None,
            ax=ax,
            scalebar_color='w',
            cutoff_vmax_data=cutoff_vmax_data if not isinstance(cutoff_vmax_data,dict) else cutoff_vmax_data[construct_label2query(construct_label)],
            **config_cropping[pair][construct_label],
            **kws_plot_image_cropped,
            test=test,
            )
        if labels_loc == 'top':
            set_label(
                    construct_label.replace('p-GFP','-GFP'),
                    x=0.5,y=1,ax=ax,
                     ha='center',
                     va='bottom',
                    color=queryprefix2color[construct_label.split('-')[0]] if construct_label.split('-')[0] in queryprefix2color and 'Delta' in construct_label else 'k',
                     )
        else:
            query,background=construct_label.replace('p-GFP','-GFP').split(' ')
            # print(query,background)
            ax.set(
                xlabel=background if background.endswith('$\Delta$') else background if show_wt else 'wild-type',
                ylabel=query if i in [0,2] else None,
            )
            ax.xaxis.set_label_position('top') 
    return axd

def plot_scatters(
    data: pd.DataFrame,
    axs: list,
    cols: list,
    ) -> list:    
    """Plot multiple scatter plots.    

    Args:
        data (pd.DataFrame): input table.
        axs (list): list of subplots.
        cols (list): columns to plot.

    Returns:
        axs (list): list of subplots.
    """
    from roux.viz.scatter import plot_scatter
    lim=data.loc[:,cols].min().min(),data.loc[:,cols].max().max()
    for ax,cols in zip(axs,list(itertools.combinations(cols,2))):
        ax=plot_scatter(
            data,
            x=cols[0],#'replicate1',
            y=cols[1],#'replicate2',
            ci=None,fit_reg=False,
            stat_kws=dict(loc=2,resample=True),
            ax=ax,
            )
        ax.set(xlim=lim,ylim=lim,)
        from roux.viz.ax_ import set_equallim
        set_equallim(ax=ax,diagonal=True)
    return axs

## functions applied on file paths
def to_plot_PCA(
    input_table_path: str,
    colors: dict,
    responsive_genes: list,
    off_axes_pad: float=0.15,
    show_arrows:bool=True,
    ax: plt.Axes=None,
    save: bool=True,
    force: bool=False,
    # test=False,
    **kws_plot_pca,
    ):
    """Plot PCA and save the output. 

    Args:
        input_table_path (str): path to the input table.
        colors (dict): colors
        responsive_genes (list): genes to mark
        off_axes_pad (float, optional): offset on the subplot margins. Defaults to 0.15.
        show_arrows (bool, optional): show arrows. Defaults to True.
        ax (plt.Axes, optional): subplot. Defaults to None.
        save (bool, optional): save the output. Defaults to True.
        force (bool, optional): overwrite the output. Defaults to False.
    """
    # df01.head(1)
    output_path=(input_table_path
                .replace(basename(dirname(dirname(input_table_path))),
                         basename(dirname(dirname(input_table_path)))+'_plots')
                .replace(splitext(input_table_path)[1],
                         '')
                )
    if exists(output_path) and not force: 
        return
    df01=read_table(input_table_path)
    logging.info(output_path)
    
    axis_labels=read_table(input_table_path.replace('/pcs/','/explained_variance/')).set_index('PC #').loc[[1,2],'explained variance'].to_dict()
    axis_labels={k:f'PC{k} ({i:.0f}%)' for k,i in axis_labels.items()}

    df1=(df01
        .loc[:,['construct label','replicate','PC #01','PC #02']]
        .rename(columns={
                    'construct label':'Label',
                     'replicate':'Replicate',
                     'PC #01':'x',
                     'PC #02':'y',
                     },
                errors='raise')
    )
    from modules.tools.viz import plot_pca
    ax=plot_pca(
        figsize=3,
        df=df1,
        colors=colors,
        use_seaborn=False,
        alpha=0.25,
        markersize=2.5,
        legend_markerscale=2.5,
        ax=ax,
        **kws_plot_pca,
    )
    ax.grid(False)
    ax.axis('off')
    ax.margins(tight=True)
    
    if not kws_plot_pca['show_legends'] and kws_plot_pca['show_n']:
        logging.info(f"cell count={len(df1)}")
        set_label(
            f"n={len(df1)}",
            x=1,y=0,
            ha='right',va='bottom',
            ax=ax,
            # fontdict=dict(fontsize=title_font_size),#20),
            )

    from roux.viz.ax_ import set_axes_minimal
    ax=set_axes_minimal(
        ax=ax,
        off_axes_pad=off_axes_pad,
        xlabel=axis_labels[1],
        ylabel=axis_labels[2],        
    )

    ## arrows only if responsive
    if show_arrows:
        ## get the centroid  
        if not 'status partner' in df01:
            df01['status partner']=df01['label'].str.split('-',expand=True)[2]
        (df01.groupby(['label','gene symbol query','status partner'])
            .agg({
                'PC #01':np.mean,
                'PC #02':np.mean,
                })
             .reset_index()
        # (df02
        .log.query(expr=f"`gene symbol query` == {responsive_genes}")
        .pivot(index='gene symbol query',
               columns=['status partner'],
               values=['PC #01','PC #02'],
              )
        .rd.flatten_columns()
        .apply(lambda x: ax.annotate("", 
                    xytext=(x['PC #01 WT'],x['PC #02 WT']), 
                    xy=(x['PC #01 DELTA'],x['PC #02 DELTA']),
                    ha='center',
                    arrowprops=dict(arrowstyle="-|>",color='k',),
                    zorder=11,
                    ),
               axis=1)
        )
    if save:
        to_plot(
            output_path,
            fmts=['pdf','png'],
            )
        plt.close(fig)
        return output_path
    else:
        return ax

def plot_dist_cutoffs(
    data: pd.Series, # series
    ##inset
    df0: pd.DataFrame=None,
    xlim_inset: list=None,
    ax: plt.Axes=None,
    **kws_hist,
    ) -> plt.Axes:
    """Plot the distributions showing thresholds.

    Args:
        data (pd.Series): input table containing columns: cutoff, label.
        df0: pd.DataFrame (pd.DataFrame, optional): data to be shown in the inset.
        xlim_inset (list, optional): x-limits for the inset.
        ax (plt.Axes, optional): subplot. Defaults to None.

    Returns:
        plt.Axes: subplot
    """
    if ax is None:
        fig,ax=plt.subplots(figsize=[8,3])
    ax=data.hist(ax=ax,**kws_hist)
    ax.set(xlabel=data.name,
           ylabel='count',
           xlim=[data.dropna().min(),data.dropna().max()]
           # xscale='log',
          )
    if not df0 is None:
        # inset axes....
        axins = ax.inset_axes([0, -1.5, 1, 1])
        data.hist(ax=axins,**kws_hist)
        for cutoff_min_cells,label in zip(df0['cutoff'],df0['cutoff label']):
            axins.axvline(x=cutoff_min_cells,
                   linestyle=':',
                   color='k',
                      )
            axins.text(x=cutoff_min_cells,
                    y=ax.get_ylim()[1],
                    s=label,
                    ha='center',
                    va='bottom'
                   )
        axins.set(
              xlabel=data.name,
               ylabel='count',
               xlim=xlim_inset if not xlim_inset is None else [df0['cutoff'].min(),df0['cutoff'].max()],
               ylim=ax.get_ylim(),
                )
        ax.indicate_inset_zoom(axins, edgecolor="gray")
    return ax

def volcano_abundance_change(
    data: pd.DataFrame,
    colx='difference between mean (DELTA-WT)',
    coly='significance\n(-log10(Q))',
    highlight: dict=None,
    protein_suffix_p=False,
    palette:list=None,
    col_text: str='gene symbol',
    ax: plt.Axes=None,
    verbose:bool=False,
    # output_path:str=None,
    # force=False,
    ) -> plt.Axes:
    """Plot abundance change as a volcano plot. 

    Args:
        data (pd.DataFrame): input table
        colx (str, optional): column with x values. Defaults to 'difference between mean (DELTA-WT)'.
        coly (str, optional): column with y values. Defaults to 'significance\n(-log10(Q))'.
        highlight (dict, optional): highlight. Defaults to None.
        protein_suffix_p (bool, optional): include protein suffix. Defaults to False.
        palette (list, optional): palette. Defaults to None.
        col_text (str, optional): column with the labels. Defaults to 'gene symbol'.
        ax (plt.Axes, optional): subplot. Defaults to None.
        verbose (bool, optional): verbose. Defaults to False.

    Returns:
        plt.Axes: subplot.
    """
    ## figure source data
    if not 'significance\n(-log10(Q))' in data:
        from roux.stat.transform import log_pval
        data=(data
            .assign(**{
                'significance\n(-log10(Q))':lambda df: log_pval(df['Q (MWU test)'],errors='ignore'),
            # 'protein label': lambda df: df['gene symbol'].str.capitalize()+'p',
            })
        )

    if ax is None:
        fig,ax=plt.subplots(figsize=[4,3])
    ax=sns.scatterplot(
        data=data.sort_values('protein abundance change',ascending=False),
        x=colx,
        y=coly,
        hue='protein abundance change',
        hue_order=['increase','decrease','ns'],
        palette=palette,
        ax=ax,
        legend=False,
        )
    # ax.legend(bbox_to_anchor=[1,1],)
    axlims=get_axlims(ax)
    ax.text(x=axlims['x']['min']+(axlims['x']['len']*0.5),
            y=-75,s="$\Delta$ background - WT background",ha='center',va='center',color='gray')
    ax.text(x=ax.get_xlim()[1],
            y=ax.get_ylim()[1],
            s="Increase $\\rightarrow$\n(compensation)",
            ha='right',va='bottom',
            color=palette[0],
           )
    ax.text(x=ax.get_xlim()[0],
            y=ax.get_ylim()[1],
            s="$\\leftarrow$ Decrease\n(dependency)",
            ha='left',va='bottom',
            color=palette[1],
           )
    # ax.axvline(x=0.2,
    #         color='gray',linestyle=':',
    #        )
    # ax.axvline(x=-0.2,
    #         color='gray',linestyle=':',
    #        )
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    from roux.stat.transform import log_pval
    ax.plot([xlim[0],-0.2,-0.2],
            [log_pval(0.05),log_pval(0.05),ylim[1]],
            color='gray',linestyle=':',
           )
    ax.plot([xlim[1],0.2,0.2],
            [log_pval(0.05),log_pval(0.05),ylim[1]],
            color='gray',linestyle=':',
           )
    if not highlight is None:
        if isinstance(highlight,int): 
            ## highlight top n
            data1=(data.sort_values(colx)
            .head(highlight)
            .append(
            data.sort_values(colx)
            .tail(highlight)
            ))        
        elif isinstance(highlight, dict):
            ## subset
            data1=data.rd.filter_rows(highlight)
        if verbose:
            print(data1)
        texts=(data1
                .apply(lambda x: ax.text(x=x[colx],
                          y=x['significance\n(-log10(Q))'],
                          s=x[col_text].capitalize()+('' if not protein_suffix_p else 'p'),
                          ),axis=1)
                .tolist()
            )
        # from adjustText import adjust_text
        # adjust_text(texts,
        #            # arrowprops=dict(arrowstyle='-', color='k'),
        #            )
        ax=sns.scatterplot(
            data=data1,
            x=colx,
            y=coly,
            ec='k',
            lw=4,
            s=50,
            fc="none",
            ax=ax,
            legend=False,
        )

    ax.set(
        xlabel='Log$_\mathrm{2}$ Fold Change (LFC)',
        ylabel='Significance\n(-log10($q$))',
        xlim=xlim,
        ylim=ylim,
    )
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    return ax

def plot_relative_abundance_change(
    df1: pd.DataFrame,
    colx: str='protein abundance change',
    xlabel: str='Protein abundance\n(log$_\mathrm{2}$)',
    coly: str='construct label',
    colxerr: str='protein abundance (log2) std',
    # delta='$\Delta$',
    colors: list=['#0065d9','#ba9e00'],
    show_significance: bool=True,
    output_path: str=None,
    ax: plt.Axes= None,
    ) -> plt.Axes:
    """Plot relative abundace for a pair.

    Args:
        df1 (pd.DataFrame): input table.
        colx (str, optional): column with x values. Defaults to 'protein abundance change'.
        xlabel (str, optional): x-axis label. Defaults to 'Protein abundance\n(log\mathrm{2}$)'.
        coly (str, optional): column with the y-values. Defaults to 'construct label'.
        colxerr (str, optional): column with the xerr values. Defaults to 'protein abundance (log2) std'.
        colors (list, optional): colors. Defaults to ['#0065d9','#ba9e00'].
        show_significance (bool, optional): show significance. Defaults to True.
        output_path (str, optional): output path. Defaults to None.
        ax (plt.Axes, optional): subplot. Defaults to None.

    Returns:
        plt.Axes: subplot.
    """
    if show_significance:
        genes_significant=df1.loc[(df1['protein abundance change']!='ns'),'gene symbol query'].tolist()
    else:
        genes_significant=[]
    df2=(df1
        .log()
        .drop(labels=['protein abundance difference (DELTA-WT)'],axis=1)
        .rd.melt_paired(suffixes=['WT','DELTA'])
        .log()
        )
    # print(df2.columns)
    # for i in [1,2]:
    #     df2[f'status gene{i}']=df2.apply(lambda x:'GFP' if x[f'gene symbol gene{i}']==x['gene symbol query'] else x['suffix'].upper(),
    #                                      axis=1)
    #     # df2[f'gene status gene{i}']=df2[f'gene status gene{i}'].fillna('GFP')
    # print(df2)
    from modules.tools.ids import to_construct_label
    from roux.lib.set import get_alt
    df2['construct label']=df2.apply(lambda x: to_construct_label(
        statuses=['GFP',x['suffix']],
        symbols=[x['gene symbol query'],
                 get_alt(x['pairs'].split('-'),x['gene symbol query']),
                ],
        sep=' ',
        fmt=True,
        show_wt=False,
        ),
        axis=1)
    
    df2=df2.sort_values(
        ['gene symbol query','suffix'],
        ascending=[True,False],
        )
    df2['protein abundance (log2) std']=df2['var'].apply(np.sqrt)
    df2[colx]=df2['mean']
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
    
    sns.barplot(
        data=df2,
        x=colx,
        y=coly,
        palette=colors,
        ax=ax,
        )
    ax.set(ylabel=None,xlabel=xlabel)
    df2['y']=df2[coly].map(get_ticklabel_position(ax=ax, axis='y'))
    df2.apply(lambda x: ax.plot(
        [x[colx]-x[colxerr],x[colx]+x[colxerr]],
        [x[coly],x[coly]],
        color='k'),
        axis=1)
    df_=(df2
        .loc[:,['gene symbol query','Q (MWU test)']]
        .drop_duplicates()
        .sort_values('gene symbol query')
        .assign(y=[0.75,0.25])
        .log.query(expr=f"`gene symbol query` == {genes_significant}")
        )
    # print(df_)
    from roux.stat.io import pval2annot
    df_.apply(lambda x: set_label(
            s=pval2annot(x['Q (MWU test)'],
            alternative='two-sided',
            fmt='<',
            replace_prefix='q',
            linebreak=True,
            ),
        x=1,#0.825,
        y=x['y'],
        ha='right',
        va='center',
        ax=ax,
        ),axis=1)
    ax.margins(x=1.2)
    ax.tick_params(axis='y', colors='k')
    
    ## group y ticklabels
    yticklabels=[t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticklabels([s.replace('-GFP','\n-GFP') for s in yticklabels])
    _scale=len(yticklabels[0].split('-')[0])
    from roux.viz.ax_ import split_ticklabels
    split_ticklabels(
        ax=ax,
        axis='y',
        fmt='group',
        sep=' ',
        group_x=(-0.173*_scale) if _scale>4 else -0.75,
        group_line_off_x=0.3 if _scale>4 else 0.19,
    )
    if not output_path is None:
        to_plot(output_path,fmts=['pdf','png'])
    else: 
        return ax

def plot_redistribution(
    data: pd.DataFrame,
    colx: str ='redistribution score',
    col_gene='gene symbol query',
    show_threshold=True,
    method_threshold: str=None,
    color_line='b',
    color_text='g',
    protein_suffixp: bool=False,
    show_annot_legend:bool=False,
    ax: plt.Axes=None,
    **kws_hist: dict,
    ) -> plt.Axes:
    """Plot redistribution scores as a histogram.

    Args:
        data (pd.DataFrame): input table.
        colx (str, optional): column with the x values. Defaults to 'redistribution score'.
        col_gene (str, optional): column with the gene names. Defaults to 'gene symbol query'.
        show_threshold (bool, optional): show the threshold. Defaults to True.
        method_threshold (str, optional): method for threshold. Defaults to None.
        color_line (str, optional): color of the line. Defaults to 'b'.
        color_text (str, optional): color of the text. Defaults to 'g'.
        protein_suffixp: add a suffix 'p' to the protein names. Defaults to False.
        show_annot_legend (bool, optional): show the annotations in legend. Defaults to False.
        ax (plt.Axes, optional): _description_. Defaults to None.

    Keyword Args:
        Keyword arguments provided to the `.hist` function.

    Returns:
        plt.Axes: subplot.
    """
    ## set data
    data['protein label']=' '+data[col_gene].str.capitalize()+('p' if protein_suffixp else '')
    ## set axes 
    if ax is None:
        fig,ax=plt.subplots(figsize=[5,4])
    from roux.viz.dist import plot_gmm,hist_annot
    from roux.viz.ax_ import get_axlims
    if method_threshold=='GMM':
        _,coff=plot_gmm(
            data.log.query(expr='control != False')[colx].copy(),
            # bins=30,
            color='k',
            show_cutoff_line=True,
            colors=[color_line,color_line,'none'],
            out_coff=True,
            hist=True,
            show_cutoff=False,
            kws_axvline=dict(color='k',linestyle=':'),
            ax=ax,
            **kws_hist,
        )
    elif show_threshold!=False:
        data.log.query(expr='control != False')[colx].hist(
            # histtype='step',
            color=color_line,
            # lw=1,
            ax=ax,
            density=False,
            **kws_hist)
        coff=show_threshold
        ax.axvline(coff,**dict(color='k',linestyle=':',ymax=0.7))
    else:
        pass
    ax.set(
        xlim=[data[colx].min(),data[colx].max()+0.5],
          )
    ax.text(
        x=coff if show_threshold==False else show_threshold,
        y=ax.get_ylim()[1]*0.75,
        s=f"$\\rightarrow${colx.split(' ')[0]}",
        color=color_text,
        )
    ax.margins(
        y=0.4,
        )
    axlims=get_axlims(ax)
    from roux.viz.annot import annot_side
    x_annot_limit=coff/axlims['x']['len']
    logging.info(f"x_annot_limit {x_annot_limit}")
    if 'control' in data:
        for i,(control_type,label) in enumerate({True:'visually classified\nas redistributed',False:'not paired with\nsister paralog'}.items()):
            # if :
            #     y_scatter=0.85*(ax.get_ylim()[1]-ax.get_ylim()[0])   
            y_scatter=(0.75 if control_type==False else 0.85) *(axlims['y']['max']-axlims['y']['min'])
            df1_=data.log.query(expr=f'control == {control_type}')
            # if f"{colx.split(' ')[0]}\n(visual classification)" in data:
            #     df1_=data.loc[(data[f"{colx.split(' ')[0]}\n(visual classification)"] == True),:]
            ax=annot_side(
                ax=ax,
                df1=df1_,
                colx=colx,
                coly= y_scatter,
                cols='protein label',
                offx3=0.05,
                length_axhline=1,
                loc='top',
                offymin=x_annot_limit if control_type==True else 0.01,
                offymax=0.99 if control_type==True else (x_annot_limit-0.05),
                scatter=True,
                scatter_alpha=1,
                text=True,
                hue='gray' if i!=0 else color_text,
                # rotation=70,
                # offx_text=-0.25,
                # offy_text=-0.4,
                va='bottom',
                kws_line=dict(lw=0.5),
                kws_scatter=dict(s=100),
                )
            if show_annot_legend:
                ax.text(
                    x=get_axlims(ax=ax)['x']['max']+(get_axlims(ax=ax)['x']['len']*0.01),
                    y=y_scatter + ((1 if i!=0 else -1) * (get_axlims(ax=ax)['y']['len']*0.05)),
                    s=f"{label}\n(n={df1_[col_gene].nunique()})",
                    ha='left',
                    va='top' if i!=0 else 'bottom',
                    # va='center',
                    color='gray' if i!=0 else color_text,
                )
    if 'protein abundance change' in data:
        for i,k in enumerate(['increase','decrease']):
            df1_=data.loc[(data['protein abundance change']==k),:]
            ax=annot_side(
                ax=ax,
                df1=df1_,
                colx=colx,
                coly=y_scatter-(0.05*(i+2)),
                cols='protein label',
                offx3=0.05,
                length_axhline=1,
                loc='top',
                offymin=0.01,
                offymax=x_annot_limit-0.05,
                scatter=True,
                text=False,
                lines=False,
                )    
            ax.text(
                x=get_axlims(ax=ax)['x']['max']+(get_axlims(ax=ax)['x']['len']*0.01),
                y=(y_scatter-(0.05*(i+2)))-0.00,
                s=f"relative {k} in abundance\n(n={df1_[col_gene].nunique()})",
                ha='left',
                va='top',
                )
    ax.set(
        ylabel='paralogs',
        )
    ax.xaxis.set_major_locator(plt.MultipleLocator(2.5))
    return ax
    
def plot_read_depth(
    data: pd.DataFrame,
    seq_ref: str,
    seq_offset_up: float,
    # gene_symbol,
    # coord,
    segment_min: float=None,
    segment_max: float=None,
    features_colors: list=[],
    axs: list=None,
    kws_bar: dict={},
    test: bool=False,
    ) -> list:
    """Plot read depth.

    Args:
        data (pd.DataFrame): input table.
        seq_ref (str): reference sequence.
        seq_offset_up (float): upstream sequence offset.
        segment_min (float, optional): segment minimum. Defaults to None.
        segment_max (float, optional): segment maximum. Defaults to None.
        features_colors (list, optional): colors of the features. Defaults to [].
        axs (list, optional): list of subplots. Defaults to None.
        kws_bar (dict, optional): kwyword arguments provided to the bar plot. Defaults to {}.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        list: list of subplots.
    """
    from dna_features_viewer import GraphicFeature, GraphicRecord
    from Bio.Seq import Seq as to_seq
    sequence = ('A'*seq_offset_up)+to_seq(seq_ref).reverse_complement()
    if (not segment_min is None) and (not segment_max is None):
        sequence = sequence[segment_min:segment_max]
    elif (not segment_min is None):
        sequence = sequence[segment_min:]
    elif (not segment_max is None):
        sequence = sequence[:segment_max]
    record = GraphicRecord(
        sequence=sequence, 
        features=[
            GraphicFeature(start=seq_offset_up, end=len(sequence), strand=+1, 
                           color=features_colors[0],
                           # linewidth=0, # border
                           linecolor='#FFFFFF',
                           label='Canonical sequence (ER)',
                          ),
            GraphicFeature(start=seq_offset_up+(19*3), end=len(sequence), strand=+1, 
                           color=features_colors[1],
                           # linewidth=0, # border
                           linecolor='#FFFFFF',
                           label='Non-canonical isoform (Cytoplasm)',
                          )
            ],
        )
    if axs is None:
        fig,axs=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=[6,1.75])
    ax, _ = record.plot(ax=axs[0])
    record = GraphicRecord(
        sequence=sequence, 
        features=[
            GraphicFeature(
                start=seq_offset_up, end=seq_offset_up+(25*3), strand=+1, 
                color=features_colors[0],
                linecolor='#FFFFFF',
                label='ER signal',
                ),
            ],
        )
    if test:
        record.plot_sequence(ax,(0,len(sequence)))
        record.plot_translation(ax, (seq_offset_up, len(sequence)), 
                                fontdict=dict(size=8,
                                             weight= 'bold'),
                                long_form_translation=False,
                               )
    ## signal peptide
    ax, _ = record.plot(ax=axs[0])
    axs[0].set(
        xticks=[],
        ylim=[-0.5,1.5],
              )
    axs[0].tick_params(
        which='both',
        top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False,labeltop=False, labelbottom=False,
        )
    # plt.setp(axs[0].get_xticklabels(), visible=False)
    # axs[0].label_outer()
    axs[0].grid(False)
    ## read depth
    (data
        .sort_values('position',ascending=False)
        .set_index('position')['depth']
        .plot.bar(
            width=1,
            **kws_bar,
            ax=axs[1]
        ))
    axs[1].set(
        xlabel=None,
        ylabel='read depth',
        xticks=[],
        xlim=[0,len(sequence)]
        )
    axs[1].xaxis.set_ticks_position('top')
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].invert_yaxis()
    # ax.figure.savefig('sequence_and_translation.png', bbox_inches='tight')
    return axs


    
def get_cropped(image, x, y, hcrop, wcrop):
    """
    Returns a cropped image within a rectangular region of interest specified by its center coordinates and dimensions.

    Args:
        image (numpy.ndarray): The input image to be cropped.
        x (int): The x-coordinate of the center of the rectangular region of interest.
        y (int): The y-coordinate of the center of the rectangular region of interest.
        hcrop (int): The height of the rectangular region of interest.
        wcrop (int): The width of the rectangular region of interest.

    Returns:
        numpy.ndarray: The cropped image within the specified region of interest.
    """
    return image[y-int(hcrop*0.5):y+int(hcrop*0.5),
                  x-int(wcrop*0.5):x+int(wcrop*0.5)]

