from typing import Literal
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_array
from anndata.tests.helpers import assert_equal
from contextlib import nullcontext

from muon import prot as pt

@pytest.fixture(params=(True, False))
def sparse(request: pytest.FixtureRequest) -> bool:
    return request.param

@pytest.fixture(params=(np.float32, np.int32))
def dtype(request: pytest.FixtureRequest) -> np.dtype | type[np.number]:
    return request.param

@pytest.fixture(params=(0, 1))
def axis(request: pytest.FixtureRequest) -> Literal[0, 1]:
    return request.param

@pytest.fixture(params=("seurat", "stoeckius", "standard"))
def flavor(request: pytest.FixtureRequest) -> Literal["seurat", "stoeckius", "standard"]:
    return request.param

@pytest.fixture(scope="session")
def data() -> np.lib.npyio.NpzFile:
    return np.load(Path(__file__).parent / "data" / "clr.npz")

@pytest.fixture
def adata(data, sparse, dtype) -> AnnData:
    x = data["x"].astype(dtype, copy=True)
    if sparse:
        x = csr_array(x)
    return AnnData(x)

@pytest.fixture
def expected_clr(data, axis, flavor) -> np.ndarray:
    return data[f"{axis}_{flavor}"]

@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:adata.X is sparse but not:UserWarning")
def test_clr(adata, expected_clr, axis, flavor):
    orig_value = adata.X.copy()
    res = pt.pp.clr(adata, axis=axis, flavor=flavor, inplace=False)
    assert_equal(orig_value, adata.X, exact=False)
    # TODO: enable when anndata 0.14 is released
    # anndata 0.13 has too strict tolerances
    # assert_equal(res.X, expected_clr)
