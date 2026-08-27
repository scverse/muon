from functools import reduce

import mudata as md
import numpy as np
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from mudata import MuData

import muon as mu


@pytest.fixture
def mdata() -> MuData:
    return MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )


def test_filter_obs_adata(mdata: MuData, rng: np.random.Generator) -> None:
    ad = mdata["mod1"]
    sub = rng.binomial(1, 0.5, ad.n_obs).astype(bool)
    mu.pp.filter_obs(ad, sub)
    assert ad.n_obs == sub.sum()


def test_filter_obs_mdata(mdata: MuData, rng: np.random.Generator) -> None:
    md = mdata.copy()
    sub = rng.binomial(1, 0.5, md.n_obs).astype(bool)
    mu.pp.filter_obs(md, sub)
    assert md.n_obs == sub.sum()
    assert md["mod1"].n_obs == mdata.obsm["mod1"][sub].sum()
    assert md["mod2"].n_obs == mdata.obsm["mod2"][sub].sum()


def test_filter_obs_mdata_consecutive(mdata: MuData, rng: np.random.Generator) -> None:
    mdata.obs["condition1"] = rng.normal(size=mdata.n_obs)
    mdata.obs["condition2"] = rng.normal(size=mdata.n_obs, scale=2)
    total_sub = np.sum((mdata.obs["condition1"] > 0) * (mdata.obs["condition2"] > 0))
    mu.pp.filter_obs(mdata, "condition1", lambda x: x > 0)
    mu.pp.filter_obs(mdata, "condition2", lambda x: x > 0)
    assert mdata.n_obs == total_sub


def test_filter_obs_mdata_consecutive_ragged(mdata: MuData, rng: np.random.Generator) -> None:
    # It should also work if data is missing in some modalities
    mod1_discard = rng.choice(range(mdata["mod1"].n_obs), size=3, replace=False)
    mod1_keep = [i for i in range(mdata["mod1"].n_obs) if i not in mod1_discard]
    md = MuData({"mod1": mdata["mod1"][mod1_keep, :].copy(), "mod2": mdata["mod2"]})

    md.obs["condition1"] = rng.normal(size=md.n_obs)
    md.obs["condition2"] = rng.normal(size=md.n_obs, scale=2)
    total_sub = np.sum((md.obs["condition1"] > 0) * (md.obs["condition2"] > 0))
    mu.pp.filter_obs(md, "condition1", lambda x: x > 0)
    mu.pp.filter_obs(md, "condition2", lambda x: x > 0)
    assert md.n_obs == total_sub


def test_filter_obs_adata_backed(mdata: MuData, rng: np.random.Generator, filepath_h5mu: str) -> None:
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

    # When backed, in-place filtering should throw a warning
    with pytest.warns(UserWarning, match="object is backed."):
        sub = rng.binomial(1, 0.5, mdata_.mod["mod1"].n_obs).astype(bool)
        print("Sub:\t", len(sub))
        print("Size:\t", mdata_.mod["mod1"].n_obs)
        mu.pp.filter_obs(mdata_.mod["mod1"], sub)


def test_filter_obs_adata_view(mdata: MuData, rng: np.random.Generator) -> None:
    pov = rng.binomial(1, 0.4, mdata.mod["mod1"].n_obs).astype(bool)
    view = mdata.mod["mod1"][pov, :]
    # When backed, in-place filtering should throw an error
    with pytest.raises(ValueError, match="does not operate on views."):
        sub = rng.binomial(1, 0.5, view.n_obs).astype(bool)
        mu.pp.filter_obs(view, sub)


def test_filter_obs_with_obsm_obsp(pbmc3k_processed: AnnData) -> None:
    A = pbmc3k_processed[:500,].copy()
    B = pbmc3k_processed[500:,].copy()
    A_subset = A[A.obs["louvain"] == "B cells"].copy()
    B_subset = B[B.obs["louvain"] == "B cells"].copy()
    mdata = mu.MuData({"A": A, "B": B}, axis=1)
    mdata.pull_obs("louvain")
    mu.pp.filter_obs(mdata, "louvain", lambda x: x == "B cells")
    assert mdata["B"].n_obs == B_subset.n_obs
    assert mdata["A"].obs["louvain"].unique() == "B cells"
    assert B.n_obs == B_subset.n_obs
    assert A.obs["louvain"].unique() == "B cells"
    assert_equal(mdata["A"], A_subset)
    assert_equal(mdata["B"], B_subset)


def test_filter_obs_with_obsm_obsp_explicit(mdata: MuData, rng: np.random.Generator) -> None:
    mdata["mod1"].obsm["X_normal"] = rng.normal(size=(mdata["mod1"].n_obs, 10))
    mdata["mod2"].obsm["X_normal"] = rng.normal(size=(mdata["mod2"].n_obs, 10))
    mdata.obsm["X_normal"] = rng.normal(size=(mdata.n_obs, 10))
    selection = mdata.obsm["X_normal"].sum(axis=1) > 0

    mdata["mod1"].obsp["connectivities"] = rng.normal(size=(mdata["mod1"].n_obs, mdata["mod1"].n_obs))
    mdata["mod2"].obsp["connectivities"] = rng.normal(size=(mdata["mod2"].n_obs, mdata["mod2"].n_obs))
    mdata.obsp["connectivities"] = rng.normal(size=(mdata.n_obs, mdata.n_obs))

    mu.pp.filter_obs(mdata, selection)
    assert mdata.n_obs == selection.sum()


def test_filter_obs_anndata(mdata: MuData, rng: np.random.Generator) -> None:
    adata = mdata["mod1"]
    adata.layers["X2"] = adata.X**2

    adata.obsm["X_normal"] = rng.normal(size=(adata.n_obs, 10))
    selection = adata.obsm["X_normal"].sum(axis=1) > 0

    adata.obsp["connectivities"] = rng.normal(size=(adata.n_obs, adata.n_obs))

    mu.pp.filter_obs(adata, selection)
    assert adata.n_obs == selection.sum()


def test_filter_var_adata(mdata: MuData, rng: np.random.Generator) -> None:
    ad = mdata["mod1"]
    sub = rng.binomial(1, 0.5, ad.n_vars).astype(bool)
    mu.pp.filter_var(ad, sub)
    assert ad.n_vars == sub.sum()


def test_filter_var_mdata(mdata: MuData, rng: np.random.Generator) -> None:
    sub = rng.binomial(1, 0.5, mdata.n_vars).astype(bool)
    sub_mod1 = mdata.varm["mod1"][sub].sum()
    sub_mod2 = mdata.varm["mod2"][sub].sum()
    mu.pp.filter_var(mdata, sub)
    assert mdata.n_vars == sub.sum()
    assert mdata["mod1"].n_vars == sub_mod1
    assert mdata["mod2"].n_vars == sub_mod2


def test_filter_var_mdata_consecutive(mdata: MuData, rng: np.random.Generator) -> None:
    mdata.var["condition1"] = rng.normal(size=mdata.n_var)
    mdata.var["condition2"] = rng.normal(size=mdata.n_var, scale=2)
    total_sub = np.sum((mdata.var["condition1"] > 0) * (mdata.var["condition2"] > 0))
    mu.pp.filter_var(mdata, "condition1", lambda x: x > 0)
    mu.pp.filter_var(mdata, "condition2", lambda x: x > 0)
    assert mdata.n_var == total_sub


def test_filter_var_adata_backed(mdata: MuData, rng: np.random.Generator, filepath_h5mu: str) -> None:
    mdata.write(filepath_h5mu)
    mdata_ = md.read_h5mu(filepath_h5mu, backed="r")
    assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

    # When backed, in-place filtering should throw a warning
    with pytest.warns(UserWarning, match="object is backed."):
        sub = rng.binomial(1, 0.5, mdata_.mod["mod1"].n_vars).astype(bool)
        print("Sub:\t", len(sub))
        print("Size:\t", mdata_.mod["mod1"].n_vars)
        mu.pp.filter_var(mdata_.mod["mod1"], sub)


def test_filter_var_adata_view(mdata: MuData, rng: np.random.Generator) -> None:
    pov = rng.binomial(1, 0.4, mdata.mod["mod1"].n_obs).astype(bool)
    view = mdata.mod["mod1"][pov, :]
    # When given a view, in-place filtering should throw an error
    with pytest.raises(ValueError, match="does not operate on views."):
        sub = rng.binomial(1, 0.5, view.n_vars).astype(bool)
        mu.pp.filter_var(view, sub)


def test_filter_var_with_varm_varp(pbmc3k_processed: AnnData, rng: np.random.Generator) -> None:
    A = pbmc3k_processed[:, :500].copy()
    B = pbmc3k_processed[:, 500:].copy()
    A_var_sel = rng.choice(np.array([0, 1]), size=A.n_vars, replace=True)
    B_var_sel = rng.choice(np.array([0, 1]), size=B.n_vars, replace=True)
    A.var["sel"] = A_var_sel
    B.var["sel"] = B_var_sel
    A_subset = A[:, A_var_sel == 1].copy()
    B_subset = B[:, B_var_sel == 1].copy()
    mdata = mu.MuData({"A": A, "B": B})
    mdata.pull_var("sel")
    mu.pp.filter_var(mdata, "sel", lambda y: y == 1)
    assert mdata.shape[1] == int(np.sum(A_var_sel) + np.sum(B_var_sel))
    assert_equal(mdata["A"], A_subset)
    assert_equal(mdata["B"], B_subset)


@pytest.mark.parametrize("empty_X", [False, True])
def test_filter_intersect_obs(mdata: MuData, rng: np.random.Generator, empty_X: bool) -> None:
    modalities = {}
    for mod, modality in mdata.mod.items():
        mod_obs_names = [f"obs{i + 1}" for i in range(modality.n_obs)]
        for obs in rng.choice(range(modality.n_obs), size=modality.n_obs // 10, replace=False):
            mod_obs_names[obs] = f"{mod}_" + str(mod_obs_names[obs])

        modalities[mod] = modality.copy()
        if empty_X:
            modalities[mod].X = None
        modalities[mod].obs_names = mod_obs_names

    mdata = MuData(modalities)

    common_obs = reduce(lambda a, b: a.intersection(b), [adata.obs_names for adata in mdata.mod.values()])

    mu.pp.intersect_obs(mdata)
    assert mdata.n_obs == len(common_obs)
    assert all(mdata.obs_names == common_obs)
