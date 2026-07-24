import pytest
import scanpy as sc  # type: ignore[import-untyped]


@pytest.fixture
def filepath_h5mu(tmp_path):
    yield str(tmp_path / "test.h5mu")


@pytest.fixture
def filepath_hdf5(tmp_path):
    yield str(tmp_path / "mofa_pytest.hdf5")


@pytest.fixture(scope="module")
def pbmc3k_processed():
    yield sc.datasets.pbmc3k_processed()
