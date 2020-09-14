import pytest

@pytest.fixture(scope='module')
def filepath_h5mu(tmpdir_factory):
	yield str(tmpdir_factory.mktemp("tmp_test_dir").join("test.h5mu"))