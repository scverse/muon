import re
import pandas as pd


def parse_region_string(region: str):
    feat_list = re.split("-|:", region)
    feature_df = pd.DataFrame(columns=["Chromosome", "Start", "End"])
    feature_df.loc[0] = feat_list
    feature_df = feature_df.astype({"Start": int, "End": int})

    return feature_df
