from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_array, sparray, spmatrix

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


def assert_allclose(x: spmatrix | sparray | np.ndarray, y: spmatrix | sparray | np.ndarray) -> None:
    if isinstance(x, spmatrix | sparray):
        x = x.toarray()
    if isinstance(y, spmatrix | sparray):
        y = y.toarray()

    np.testing.assert_allclose(x, y)


def assert_allequal(x: spmatrix | sparray | np.ndarray, y: spmatrix | sparray | np.ndarray) -> None:
    if isinstance(x, spmatrix | sparray):
        x = x.toarray()
    if isinstance(y, spmatrix | sparray):
        y = y.toarray()

    assert np.all(x == y)


def test_tfidf(adata, expected_tfidf, log_tf, log_idf):
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf)
    assert_allclose(adata.X, expected_tfidf)


def test_tfidf_view(adata, expected_tfidf, log_tf, log_idf):
    view = adata[:, :]
    ac.pp.tfidf(view, log_tf=log_tf, log_idf=log_idf)
    assert_allclose(view.X, expected_tfidf)


def test_tfidf_copy(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    copy = ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, copy=True)
    assert_allequal(orig_value, adata.X)
    assert_allclose(copy.X, expected_tfidf)


def test_tfidf_inplace(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    res = ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, inplace=False)
    assert_allequal(orig_value, adata.X)
    assert_allclose(res, expected_tfidf)


def test_tfidf_to_layer(adata, expected_tfidf, log_tf, log_idf):
    orig_value = adata.X.copy()
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, to_layer="new")
    assert_allequal(orig_value, adata.X)
    assert_allclose(adata.layers["new"], expected_tfidf)


def test_tfidf_from_layer(adata, expected_tfidf, log_tf, log_idf):
    adata.layers["counts"] = adata.X
    adata.X = None
    ac.pp.tfidf(adata, log_tf=log_tf, log_idf=log_idf, from_layer="counts")
    assert_allclose(adata.X, expected_tfidf)
