from pathlib import Path

import h5py
import numpy as np
import pytest
from anndata import AnnData
from mudata import MuData
from scipy import sparse

import muon as mu


@pytest.fixture
def filepath_hdf5(tmp_path: Path) -> str:
    return str(tmp_path / "mofa.hdf5")


@pytest.fixture
def mdata(rng: np.random.Generator) -> MuData:
    z = rng.normal(size=(100, 5))
    w1 = rng.normal(size=(90, 5))
    w2 = rng.normal(size=(50, 5))
    e1 = rng.normal(size=(100, 90))
    e2 = rng.normal(size=(100, 50))
    y1 = z @ w1.T + e1
    y2 = z @ w2.T + e2
    mdata = MuData({"y1": AnnData(y1), "y2": AnnData(y2)})
    mdata.obs["group"] = rng.choice((0, 1), size=mdata.n_obs)
    mdata.obs["group_cat"] = mdata.obs["group"].astype("category")
    mdata["y1"].obs["group"] = rng.choice((0, 1), size=mdata["y1"].n_obs)
    mdata["y1"].obs["group_cat"] = mdata["y1"].obs["group"].astype("category")
    return mdata


def test_mofa_nfactors(mdata: MuData, filepath_hdf5: str) -> None:
    n_factors = 10
    mu.tl.mofa(mdata, n_factors=n_factors, quiet=True, verbose=False, outfile=filepath_hdf5)

    # Only first 5 factors should have high R2
    for view_r2 in mdata.uns["mofa"]["variance"].values():
        assert np.all(view_r2[:5] > 5)
        assert np.all(view_r2[5:] <= 5)


def test_mofa_anndata(mdata: MuData, filepath_hdf5: str) -> None:
    mu.tl.mofa(mdata["y1"], n_factors=10, quiet=True, verbose=False, outfile=filepath_hdf5)
    assert "X_mofa" in mdata["y1"].obsm
    assert "LFs" in mdata["y1"].varm


@pytest.mark.parametrize("group_col", ("group", "group_cat"))
@pytest.mark.parametrize("use_adata", (False, True))
def test_mofa_groups(
    mdata: MuData, group_col: str, use_adata: bool, rng: np.random.Generator, filepath_hdf5: str
) -> None:
    if use_adata:
        mdata = mdata["y1"]
    mu.tl.mofa(
        mdata,
        groups_label=group_col,
        n_factors=10,
        scale_views=False,
        scale_groups=False,
        center_groups=False,
        quiet=True,
        verbose=False,
        outfile=filepath_hdf5,
    )
    assert "X_mofa" in mdata.obsm
    assert "LFs" in mdata.varm

    groups = np.sort(mdata.obs[group_col].unique())
    groups_str = groups.astype(str)
    with h5py.File(filepath_hdf5) as mofa_out:
        assert np.all(np.asarray(sorted(mofa_out["samples_metadata"].keys())) == groups_str)
        for group, group_str in zip(groups, groups_str, strict=True):
            submdata = mdata[mdata.obs[group_col] == group]
            if use_adata:
                assert np.all(mofa_out["data"]["data"][group_str][()] == submdata.X)
            else:
                for view_name, view in submdata.mod.items():
                    assert np.all(mofa_out["data"][view_name][group_str][()] == view.X)
            assert np.all(mofa_out["samples_metadata"][group_str][group_col][()] == group)
            assert np.all(mofa_out["expectations"]["Z"][group_str][()].T == submdata.obsm["X_mofa"])


@pytest.mark.parametrize("sparsity", (0, 1, 2))
def test_mofa_obs_union(mdata, sparsity: int, filepath_hdf5: str) -> None:
    y1 = mdata["y1"]
    y2 = mdata["y2"]
    if sparsity in (0, 2):
        y1.X = sparse.csr_matrix(y1.X)
    if sparsity in (1, 2):
        y2.X = sparse.csr_matrix(y2.X)
    mdata = MuData({"y1": y1[:-10], "y2": y2[10:]})
    mu.tl.mofa(mdata, n_factors=10, quiet=True, verbose=False, use_obs="union", outfile=filepath_hdf5)
    assert "X_mofa" in mdata.obsm
    assert "LFs" in mdata.varm
