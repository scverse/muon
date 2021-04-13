from collections import Counter
import pandas as pd
import numpy as np
from anndata.utils import make_index_unique


def _make_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    dup_idx = np.zeros((df.shape[0],), dtype=np.uint8)
    if not df.index.is_unique:
        duplicates = np.nonzero(df.index.duplicated())[0]
        cnt = Counter()
        for dup in duplicates:
            idxval = df.index[dup]
            newval = cnt[idxval] + 1
            dup_idx[dup] = newval
            cnt[idxval] = newval
    return df.set_index(dup_idx, append=True)


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(level=-1, drop=True)
