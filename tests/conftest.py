import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def filepath_h5mu(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("tmp_test_dir") / "test.h5mu")


@pytest.fixture(scope="module")
def filepath_hdf5(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("tmp_mofa_dir") / "mofa_pytest.hdf5")


@pytest.fixture(scope="module")
def pbmc3k_processed() -> AnnData:
    return sc.datasets.pbmc3k_processed()
