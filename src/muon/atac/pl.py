import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from mudata import MuData
from scipy.sparse import issparse, sparray, spmatrix

from . import tl


def _to_dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if issparse(x) else x)


def _average_peaks(
    adata: AnnData, keys: Iterable[str], average: str | None, func: str, use_raw: bool, layer: str | None
):
    # Define the function to be used for aggregation
    if average:
        avg_func = getattr(np, func)

    matrix = adata.layers[layer] if layer else adata.raw.X if use_raw else adata.X
    if matrix is None:
        raise ValueError("No count matrix found in .X. Provide a `layer` or set `use_raw=True`.")

    # New keys will be placed here
    attr_names = []
    tmp_names = []
    x = pd.DataFrame(index=adata.obs.index)
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
                warnings.warn(f"Peaks for {key} are not found.", stacklevel=2)
                continue

            peaksidx = adata.var.index.get_indexer_for(peaks)

            if average == "total" or average == "all":
                attr_name = f"{key} (all peaks)"
                attr_names.append(attr_name)
                tmp_names.append(attr_name)

                if attr_name not in adata.obs.columns:
                    x[attr_name] = np.asarray(avg_func(matrix[:, peaksidx], axis=1)).reshape(-1)

            elif average == "peak_type":
                peak_types = peak_sel.peak_type

                # {'promoter': ['chrX:NNN_NNN', ...], 'distal': ['chrX:NNN_NNN', ...]}
                peak_dict = defaultdict(list)
                for k, v in zip(peak_types, peaksidx, strict=True):
                    peak_dict[k].append(v)

                # 'CD4 (promoter peaks)', 'CD4 (distal peaks)'
                for t, p in peak_dict.items():
                    attr_name = f"{key} ({t} peaks)"
                    attr_names.append(attr_name)
                    tmp_names.append(attr_name)

                    if attr_name not in adata.obs.columns:
                        x[attr_name] = np.asarray(avg_func(matrix[:, np.asarray(p)], axis=1)).reshape(-1)

            else:
                # No averaging, one plot per peak
                if average is not None and average is not False and average != -1:
                    warnings.warn(
                        f"Plotting individual peaks since {average} was not recognised. Try using 'total' or 'peak_type'.",
                        stacklevel=2,
                    )
                attr_names += list(peaks.values)
                x_peaks = pd.DataFrame(_to_dense(matrix[:, peaksidx]), columns=peaks.values, index=x.index)
                x = pd.concat([x, x_peaks], axis=1)

        else:
            attr_names.append(key)
            keyloc = adata.var.index.get_loc(key)
            x[key] = _to_dense(matrix[:, keyloc]).reshape(-1)

    return (x, attr_names, tmp_names)


def embedding(
    data: AnnData | MuData,
    basis: str,
    color: str | list[str] | None = None,
    average: str | None = "total",
    func: str = "mean",
    use_raw: bool = True,
    layer: str | None = None,
    **kwargs,
) -> Axes | list[Axes] | None:
    """Scatter plot in the define basis.

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
        retval = sc.pl.embedding(ad, basis=basis, color=attr_names, **kwargs)
        for aname in attr_names:
            try:
                adata.uns[f"{aname}_colors"] = ad.uns[f"{aname}_colors"]
            except KeyError:
                pass
        return retval

    else:
        return sc.pl.embedding(adata, basis=basis, use_raw=use_raw, layer=layer, **kwargs)


def pca(data: AnnData | MuData, **kwargs) -> Axes | list[Axes] | None:
    """Scatter plot for principle components.

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="pca", **kwargs)


def lsi(data: AnnData | MuData, **kwargs) -> Axes | list[Axes] | None:
    """Scatter plot for latent semantic indexing components.

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="lsi", **kwargs)


def umap(data: AnnData | MuData, **kwargs) -> Axes | list[Axes] | None:
    """Scatter plot in UMAP space.

    See sc.pl.embedding for details.
    """
    return embedding(data, basis="umap", **kwargs)


def mofa(mdata: MuData, **kwargs) -> Axes | list[Axes] | None:
    """Scatter plot in MOFA factors coordinates.

    See sc.pl.embedding for details.
    """
    return embedding(mdata, "mofa", **kwargs)


def dotplot(
    data: AnnData | MuData,
    var_names: str | Sequence[str] | Mapping[str, str | Sequence[str]],
    groupby: str | None = None,
    average: str | None = "total",
    func: str = "mean",
    use_raw: bool = False,
    layer: str | None = None,
    **kwargs,
) -> None:
    """Dotplot.

    See sc.pl.embedding for details.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    keys: Iterable[str]
    if isinstance(var_names, str):
        keys = [var_names]
    elif isinstance(var_names, Iterable):
        keys = var_names
    else:
        raise TypeError("Expected var_names to be a string or an iterable.")

    x, attr_names, tmp_names = _average_peaks(
        adata=adata, keys=keys, average=average, func=func, use_raw=use_raw, layer=layer
    )
    ad = AnnData(x, obs=adata.obs)
    sc.pl.dotplot(ad, var_names=attr_names, groupby=groupby, **kwargs)

    return None


def tss_enrichment(
    data: AnnData, color: str | Sequence[str] | None = None, title: str = "TSS Enrichment", ax: Axes | None = None
) -> None:
    """Plot relative enrichment scores around a TSS.

    Args:
        data: AnnData object with cell x TSS_position matrix as generated by `muon.atac.tl.tss_enrichment`.
        color: Column name of .obs slot of the AnnData object which to group TSS signals by.
        title: Plot title.
        ax: A matplotlib axes object.
    """
    ax = ax or plt.gca()

    if color is not None:
        if isinstance(color, str):
            color = [color]

        obs: pd.DataFrame = data.obs
        for name, group in obs.groupby(color):
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
    X: np.ndarray | spmatrix | sparray = data.X
    means = X.mean(axis=0)
    ax.plot(x, means, **kwargs)
    if sd:
        stddev = np.sqrt(X.var(axis=0))
        plt.fill_between(x, means - stddev, means + stddev, alpha=0.2)


def fragment_histogram(
    data: AnnData | MuData,
    region: str = "chr1-1-2000000",
    groupby: str | Sequence[str] | None = None,
    barcodes: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
) -> None:
    """Plot Histogram of Fragment lengths within specified region.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        region: Region to plot. Specified with the format `chr1:1-2000000` or`chr1-1-2000000`.
        groupby: Column name(s) of .obs slot of the AnnData object according to which the plot is split.
        barcodes: Column name of .obs slot of the AnnData object
            with barcodes corresponding to the ones in the fragments file.
        show: Show the plot, do not return axis.
        save: If `True` or a `str`, save the figure.
            A string is appended to the default filename.
            Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    """
    from scanpy.plotting._utils import savefig_or_show

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    fragment_path = adata.uns["files"]["fragments"]
    fragments = tl.fetch_regions_to_df(fragment_path=fragment_path, features=region)

    fragments["length"] = fragments.End - fragments.Start
    fragments.set_index(keys="Cell", inplace=True)
    obs: pd.DataFrame = adata.obs
    if barcodes and barcodes in obs.columns:
        fragments = fragments.join(obs.set_index(barcodes), how="right")
    else:
        fragments = fragments.join(obs, how="right")

    # Handle sns.distplot deprecation and sns.histplot addition
    hist = sns.histplot if hasattr(sns, "histplot") else sns.distplot

    binwidth = 5
    if hasattr(sns, "histplot"):
        kwargs = {"binwidth": binwidth}
    else:
        n_bins = int(np.ceil(fragments.length.max() / binwidth))
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

    savefig_or_show("fragment_histogram_", show=show, save=save)
