from collections import defaultdict
from typing import Union, Optional, List, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from .._core.mudata import MuData
from . import tools


def _average_peaks(
    adata: AnnData,
    keys: List[str],
    average: Optional[str],
    func: str,
    use_raw: bool,
    layer: Optional[str],
):
    # Define the function to be used for aggregation
    if average:
        avg_func = getattr(np, func)
    # New keys will be placed here
    attr_names = []
    tmp_names = []
    x = adata.obs.loc[:, []]
    for key in keys:
        if key not in adata.var_names and key not in adata.obs.columns:
            if "atac" not in adata.uns or "peak_annotation" not in adata.uns["atac"]:
                raise KeyError(
                    f"There is no feature or feature annotation {key}. If it is a gene name, load peak annotation with muon.atac.pp.add_peak_annotation first."
                )
            peak_sel = adata.uns["atac"]["peak_annotation"].loc[[key]]

            # only use peaks that are in the object (e.g. haven't been filtered out)
            peak_sel = peak_sel[peak_sel.peak.isin(adata.var_names.values)]

            peaks = peak_sel.peak

            if len(peaks) == 0:
                warnings.warn(f"Peaks for {key} are not found.")
                continue

            if average == "total" or average == "all":
                attr_name = f"{key} (all peaks)"
                attr_names.append(attr_name)
                tmp_names.append(attr_name)

                if attr_name not in adata.obs.columns:
                    if layer:
                        x[attr_name] = np.asarray(
                            avg_func(adata[:, peaks].layers[layer], axis=1)
                        ).reshape(-1)
                    elif use_raw:
                        x[attr_name] = np.asarray(avg_func(adata[:, peaks].raw.X, axis=1)).reshape(
                            -1
                        )
                    else:
                        x[attr_name] = np.asarray(avg_func(adata[:, peaks].X, axis=1)).reshape(-1)

            elif average == "peak_type":
                peak_types = peak_sel.peak_type

                # {'promoter': ['chrX:NNN_NNN', ...], 'distal': ['chrX:NNN_NNN', ...]}
                peak_dict = defaultdict(list)
                for k, v in zip(peak_types, peaks):
                    peak_dict[k].append(v)

                # 'CD4 (promoter peaks)', 'CD4 (distal peaks)'
                for t, p in peak_dict.items():
                    attr_name = f"{key} ({t} peaks)"
                    attr_names.append(attr_name)
                    tmp_names.append(attr_name)

                    if attr_name not in adata.obs.columns:
                        if layer:
                            x[attr_name] = np.asarray(
                                avg_func(adata[:, p].layers[layer], axis=1)
                            ).reshape(-1)
                        elif use_raw:
                            x[attr_name] = np.asarray(avg_func(adata[:, p].raw.X, axis=1)).reshape(
                                -1
                            )
                        else:
                            x[attr_name] = np.asarray(avg_func(adata[:, p].X, axis=1)).reshape(-1)

            else:
                # No averaging, one plot per peak
                if average is not None and average is not False and average != -1:
                    warnings.warn(
                        f"Plotting individual peaks since {average} was not recognised. Try using 'total' or 'peak_type'."
                    )
                attr_names += list(peaks.values)
                if layer:
                    x_peaks = adata[:, peaks].layers[layer]
                elif use_raw:
                    x_peaks = adata[:, peaks].raw.X
                else:
                    x_peaks = adata[:, peaks].X
                if issparse(x_peaks):
                    x_peaks = x_peaks.toarray()
                x_peaks = pd.DataFrame(np.asarray(x_peaks), columns=peaks.values, index=x.index)
                x = pd.concat([x, x_peaks], axis=1)

        else:
            attr_names.append(key)
            if layer:
                x_peak = adata[:, key].layers[layer]
            elif use_raw:
                x_peak = adata[:, key].raw.X
            else:
                x_peak = adata[:, key].X
            if issparse(x_peak):
                x_peak = x_peak.toarray()
            x_peak = x_peak.reshape(-1)
            x[key] = x_peak

    return (x, attr_names, tmp_names)


def embedding(
    data: Union[AnnData, MuData],
    basis: str,
    color: Optional[Union[str, List[str]]] = None,
    average: Optional[str] = "total",
    func: Optional[str] = "mean",
    use_raw: bool = True,
    layer: Optional[str] = None,
    **kwargs,
):
    """
    Scatter plot in the define basis

    See sc.pl.embedding for details.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if color is not None:
        if isinstance(color, str):
            keys = [color]
        elif isinstance(color, Iterable):
            keys = color
        else:
            raise TypeError("Expected color to be a string or an iterable.")

        x, attr_names, _ = _average_peaks(
            adata=adata, keys=keys, average=average, func=func, use_raw=use_raw, layer=layer
        )
        ad = AnnData(x, obs=adata.obs, obsm=adata.obsm)
        return sc.pl.embedding(ad, basis=basis, color=attr_names, **kwargs)

    else:
        return sc.pl.embedding(adata, basis=basis, use_raw=use_raw, layer=layer, **kwargs)

    return None


def pca(data: Union[AnnData, MuData], **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot for principle components

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="pca", **kwargs)


def lsi(data: Union[AnnData, MuData], **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot for latent semantic indexing components

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="lsi", **kwargs)


def umap(data: Union[AnnData, MuData], **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in UMAP space

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="umap", **kwargs)


def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in MOFA factors coordinates

    See sc.pl.embedding for details.
    """
    return embedding(mdata, "mofa", **kwargs)


def dotplot(
    data: Union[AnnData, MuData],
    var_names: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]],
    groupby: Optional[Union[str]] = None,
    average: Optional[str] = "total",
    func: Optional[str] = "mean",
    use_raw: Optional[Union[bool]] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """
    Dotplot

    See sc.pl.embedding for details.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if isinstance(var_names, str):
        keys = [var_names]
    elif isinstance(var_names, Iterable):
        keys = var_names
    else:
        raise TypeError("Expected var_names to be a string or an iterable.")

    x, attr_names, tmp_names = _average_peaks(
        adata=adata,
        keys=keys,
        average=average,
        func=func,
        use_raw=use_raw,
        layer=layer,
    )
    ad = AnnData(x, obs=adata.obs)
    sc.pl.dotplot(ad, var_names=attr_names, groupby=groupby, **kwargs)

    return None


def tss_enrichment(
    data: AnnData,
    color: Optional[Union[str]] = None,
    title: str = "TSS Enrichment",
    ax: Optional[Axes] = None,
):

    """
    Plot relative enrichment scores around a TSS.

    Parameters
    ----------
    data
        AnnData object with cell x TSS_position matrix as generated by `muon.atac.tl.tss_enrichment`.
    color
        Column name of .obs slot of the AnnData object which to group TSS signals by.
    title
        Plot title.
    ax
        A matplotlib axes object.
    """
    ax = ax or plt.gca()

    if color is not None:
        if isinstance(color, str):
            color = [color]

        groups = data.obs.groupby(color)

        for name, group in groups:
            ad = data[group.index]
            _tss_enrichment_single(ad, ax, label=name)
    else:
        _tss_enrichment_single(data, ax)

    # TODO Not sure how to best deal with plot returning/showing
    ax.set_title(title)
    ax.set_xlabel("Distance from TSS, bp")
    ax.set_ylabel("Average TSS enrichment score")
    if color:
        ax.legend(loc="upper right", title=", ".join(color))
    plt.show()
    return None


def _tss_enrichment_single(data: AnnData, ax: Axes, sd: bool = False, *args, **kwargs):
    x = data.var["TSS_position"]
    means = data.X.mean(axis=0)
    ax.plot(x, means, **kwargs)
    if sd:
        sd = np.sqrt(data.X.var(axis=0))
        plt.fill_between(
            x,
            means - sd,
            means + sd,
            alpha=0.2,
        )


def fragment_histogram(
    data: Union[AnnData, MuData],
    region: str = "chr1-1-2000000",
    groupby: Optional[Union[str]] = None,
):
    """
    Plot Histogram of Fragment lengths within specified region.
    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    region
        Region to plot. Specified with the format `chr1:1-2000000` or`chr1-1-2000000`.
    groupby
        Column name(s) of .obs slot of the AnnData object according to which the plot is split.
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    fragment_path = adata.uns["files"]["fragments"]
    fragments = tools.fetch_regions_to_df(fragment_path=fragment_path, features=region)

    fragments["length"] = fragments.End - fragments.Start
    fragments.set_index(keys="Cell", inplace=True)
    fragments = fragments.join(adata.obs, how="right")

    # Handle sns.distplot deprecation and sns.histplot addition
    hist = sns.histplot if hasattr(sns, "histplot") else sns.distplot

    binwidth = 5
    if hasattr(sns, "histplot"):
        kwargs = {"binwidth": binwidth}
    else:
        n_bins = np.int(np.ceil(fragments.length.max() / binwidth))
        kwargs = {"bins": n_bins, "kde": False}

    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        if len(groupby) > 2:
            raise ValueError("Maximum 2 categories in groupby")
        elif len(groupby) == 2:
            g = sns.FacetGrid(fragments, col=groupby[0], row=groupby[1], sharey=False)
        elif len(groupby) == 1:
            g = sns.FacetGrid(fragments, col=groupby[0], sharey=False)
        g.map(hist, "length", **kwargs)
        g.set_xlabels("Fragment length (bp)")
    else:
        # Handle sns.distplot deprecation and sns.histplot addition
        g = hist(fragments.length, **kwargs)
        g.set_xlabel("Fragment length (bp)")
    g.set(xlim=(0, 1000))

    plt.show()

    return None
