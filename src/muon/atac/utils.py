import re

import pandas as pd  # type: ignore[import-untyped]


def parse_region_string(region: str) -> pd.DataFrame:
    """Parse a region string such as ``chr1:100-200`` into a Chromosome/Start/End data frame."""
    feat_list = re.split("-|:", region)
    feature_df = pd.DataFrame(columns=["Chromosome", "Start", "End"])
    feature_df.loc[0] = feat_list
    feature_df = feature_df.astype({"Start": int, "End": int})

    return feature_df
