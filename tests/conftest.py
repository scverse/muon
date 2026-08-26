import numpy as np
import pytest
import scanpy as sc


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def filepath_h5mu(tmp_path_factory):
    yield str(tmp_path_factory.mktemp("tmp_test_dir") / "test.h5mu")


@pytest.fixture(scope="module")
def filepath_hdf5(tmp_path_factory):
    yield str(tmp_path_factory.mktemp("tmp_mofa_dir") / "mofa_pytest.hdf5")


@pytest.fixture(scope="module")
def pbmc3k_processed():
    yield sc.datasets.pbmc3k_processed()
