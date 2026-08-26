from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from scipy.sparse import csr_array

from muon import atac as ac


@pytest.fixture(params=[True, False])
def sparse(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False])
def log_tf(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False])
def log_idf(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(scope="session")
def data() -> np.lib.npyio.NpzFile:
    return np.load(Path(__file__).parent / "data" / "tfidf.npz")


@pytest.fixture
def adata(data, sparse) -> AnnData:
    if sparse:
        x = csr_array(data["x"])
    else:
        x = data["x"].copy()
    return AnnData(x)


@pytest.fixture
def expected_tfidf(data, log_tf, log_idf) -> np.ndarray:
    return data[f"{log_tf}_{log_idf}"]


def test_tfidf(adata, expected_tfidf, log_tf, log_idf):
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf)
    assert_equal(adata.X, expected_tfidf)


def test_tfidf_view(adata, expected_tfidf, log_tf, log_idf):
    view = adata[:, :]
    ac.pp.tfidf(view, log_tf=log_tf, log_idf=log_idf)
    assert_equal(view.X, expected_tfidf)


def test_tfidf_copy(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    copy = ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, copy=True)
    assert_equal(orig_value, adata.X, exact=True)
    assert_equal(copy.X, expected_tfidf)


def test_tfidf_inplace(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    res = ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, inplace=False)
    assert_equal(orig_value, adata.X, exact=True)
    assert_equal(res, expected_tfidf)


def test_tfidf_to_layer(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, to_layer="new")
    assert_equal(orig_value, adata.X, exact=True)
    assert_equal(adata.layers["new"], expected_tfidf)


def test_tfidf_from_layer(adata, expected_tfidf, log_tf, log_idf):
    adata.layers["counts"] = adata.X
    adata.X = None
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, from_layer="counts")
    assert_equal(adata.X, expected_tfidf)
