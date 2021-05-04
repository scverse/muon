import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

# import pandas as pd
from . import utils
from . import tools
from typing import Union, Optional
from .._core.mudata import MuData
from anndata import AnnData
from coolbox.core.track import Track
from coolbox.core.track import GTF

# def colorrug(x, ax, column, color_dict):
#     """Used to plot categorical variables in .obs on the side of a plot"""
#     for lin in pd.unique(x.stage_lineage):
#         sns.rugplot(
#             y=pd.unique(x[x.stage_lineage == lin].pseudotime),
#             ax=ax,
#             color=color_dict[lin],
#             alpha=0.9,
#         )
class MetData(Track):
    def __init__(self, data: Union[AnnData, MuData]):
        self.adata = utils.get_modality(data, modality="met")
        super().__init__({"height": 8})  # init Track class

    def fetch_data(self, genome_range, **kwargs):
        region_df = tools.fetch_region(self.adata, genome_range, return_region=True)
        return region_df

    def plot(self, ax, genome_range, param_dict=None, yaxis="pseudotime", **kwargs):
        """
        Plots individual methylation events as a scatterplot with genome as x axis and a continuous variable as y axis. Requires an active region in .uns['active_region']
        Parameters
        ----------
        data
                AnnData object with peak counts or multimodal MuData object with 'met' modality.
        """

        # ax = ax or plt.gca()
        region_df = self.fetch_data(genome_range)
        region_df["rate"] = region_df["rate"] > 0.5
        params = {"pointstyles": {1: "o", 0: "x"}}
        if param_dict:
            params.update(param_dict)

        if yaxis not in region_df.columns:
            if yaxis in self.adata.obs.columns:
                region_df = region_df.merge(self.adata.obs, how="inner")
            else:
                raise ValueError(
                    f"Could not find column {yaxis} in .obs or in the active region. Please specify a different yaxis."
                )
        colors = {
            1: (0.5, 0.0, 0.0, 1.0),
            0: (0.0, 0.0, 0.5, 1.0),
        }  # Extremes of the jet colorscale
        plotargs = {
            "hue": "rate",
            "palette": colors,
            "marker": "s",
            "s": 4,
            "edgecolor": None,
            "alpha": 0.3,
        }
        # plotargs.update(kwargs)
        print(plotargs)
        sns.scatterplot(x="pos", y=yaxis, data=region_df, ax=ax, **plotargs)
        # ax.scatter(region_df.pos, region_df[yaxis])
        ax.set_xlim(genome_range.start, genome_range.end)
        # return region_df

        # for met, pts in pointstyles.items():
        #     mask = y[:, 0] == met
        #     sc = ax.plot(X[mask, -1], X[mask, 0], marker=pts, ls='', ms=3, color = colors[met]**params)


from dna_features_viewer import GraphicFeature, GraphicRecord
from coolbox.utilities import GenomeRange
import random
import re


class EnsemblGTF(GTF):
    def plot(self, ax, gr: GenomeRange, **kwargs):
        self.ax = ax
        df = self.fetch_plot_data(gr)
        df["feature_name"] = df["attribute"].str.extract("[Parent|ID]=(.*?);").iloc[:, 0]
        if self.has_prop("row_filter"):
            print("EnsemblGTF does not make use of row_filter. Please use row_filter_func instead.")
        #     filters = self.properties["row_filter"]
        #     for filter_ in filters.split(";"):
        #         try:
        #             op_idx = list(re.finditer("[=><!]", filter_))[0].start()
        #             l_ = filter_[:op_idx].strip()
        #             r_ = filter_[op_idx:]
        #             print(l_, "and", r_)
        #             df = eval(f'df[df["{l_}"]{r_}]')
        #         except IndexError:
        #             raise ValueError(f"row filter {filter_} is not valid.")
        # print(df)
        if self.has_prop("row_filter_func"):
            func = self.properties["row_filter_func"]
            df = func(df)
        # print(df)
        region_length = gr.end - gr.start
        len_ratio_th = self.properties["length_ratio_thresh"]
        df = df[(df["end"] - df["start"]) > region_length * len_ratio_th]
        features = []
        for _, row in df.iterrows():
            gf = GraphicFeature(
                start=row["start"],
                end=row["end"],
                strand=(1 if row["strand"] == "+" else -1),
                label=row["feature_name"],
                color=random.choice(self.colors),
            )
            features.append(gf)
        record = GraphicRecord(
            sequence_length=gr.end - gr.start, features=features, first_index=gr.start
        )
        record.plot(ax=ax, with_ruler=False, draw_line=False)
        self.plot_label()


def colorrug(x, ax, color_dict):
    for lin in pd.unique(x.stage_lineage):
        sns.rugplot(
            y=pd.unique(x[x.stage_lineage == lin].pseudotime),
            ax=ax,
            height=0.1,
            color=color_dict[lin],
            expand_margins=False,
            legend=True,
            alpha=0.9,
        )
