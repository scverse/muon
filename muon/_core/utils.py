import pandas as pd
from anndata.utils import make_index_unique


def _make_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(make_index_unique(df.index), append=True)


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(level=-1, drop=True)
