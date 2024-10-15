import pytest
import scanpy as sc

@pytest.fixture(scope="module")
def filepath_h5mu(tmpdir_factory):
    yield str(tmpdir_factory.mktemp("tmp_test_dir").join("test.h5mu"))


@pytest.fixture(scope="module")
def filepath_hdf5(tmpdir_factory):
    yield str(tmpdir_factory.mktemp("tmp_mofa_dir").join("mofa_pytest.hdf5"))

@pytest.fixture(scope="module")
def pbmc3k_processed():
    yield sc.datasets.pbmc3k_processed()
