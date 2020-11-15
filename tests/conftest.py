import pytest


@pytest.fixture(scope="module")
def filepath_h5mu(tmpdir_factory):
    yield str(tmpdir_factory.mktemp("tmp_test_dir").join("test.h5mu"))


@pytest.fixture(scope="module")
def filepath_hdf5(tmpdir_factory):
    yield str(tmpdir_factory.mktemp("tmp_mofa_dir").join("mofa_pytest.hdf5"))
