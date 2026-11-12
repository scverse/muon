"""Tests for the HoloViews interface for MuData (``muon._hv``)."""

from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from mudata import MuData

from muon import _hv


@pytest.fixture
def mdata() -> MuData:
    rng = np.random.default_rng(0)
    n = 30
    obs_names = [f"cell{i}" for i in range(n)]
    mods = {}
    for name, n_vars in (("mod1", 4), ("mod2", 3)):
        mod = AnnData(X=rng.normal(size=(n, n_vars)).astype("float32"))
        mod.obs_names = obs_names
        mod.var_names = [f"{name}_g{j}" for j in range(n_vars)]
        mod.obsm["umap"] = rng.normal(size=(n, 2))
        mods[name] = mod
    mdata = MuData(mods)
    mdata.obs["condition"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    mdata.obs["score"] = rng.normal(size=n)
    mdata.obsm["X_joint"] = rng.normal(size=(n, 2))
    mdata.obsm["umap"] = rng.normal(size=(n, 2))
    return mdata


@pytest.fixture
def hv() -> ModuleType:
    """HoloViews, with the hv-anndata interface registered."""
    hv = pytest.importorskip("holoviews")
    hv_anndata = pytest.importorskip("hv_anndata")
    hv_anndata.register()  # what `sc.pl.hv_init()` does for AnnData
    pytest.importorskip("bokeh")
    hv.extension("bokeh")
    return hv


@pytest.fixture
def scanpy_v2() -> Iterator[None]:
    """Activate scanpy's ScanpyV2Preview preset for the duration of a test."""
    if not hasattr(sc, "Preset"):
        pytest.skip("this scanpy version has no presets")
    old = sc.settings.preset
    try:
        sc.settings.preset = sc.Preset.ScanpyV2Preview
    except ImportError as e:
        pytest.skip(f"the preset needs optional dependencies: {e}")
    yield
    sc.settings.preset = old


# --- which preset is active? --------------------------------------------------------------------------


def test_default_preset_is_not_v2() -> None:
    assert not _hv._scanpy_preset_is_v2()


@pytest.mark.skipif(not hasattr(sc, "Preset"), reason="this scanpy version has no presets")
def test_preset_is_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "settings", SimpleNamespace(preset=sc.Preset.ScanpyV2Preview))
    assert _hv._scanpy_preset_is_v2()


def test_scanpy_without_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sc, "Preset", raising=False)
    assert not _hv._scanpy_preset_is_v2()


# --- reading a MuData ---------------------------------------------------------------------------------


def test_scatter(hv: ModuleType, mdata: MuData) -> None:
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1]])
    assert len(plot) == mdata.n_obs
    np.testing.assert_allclose(plot.dimension_values(0), mdata.obsm["X_joint"][:, 0])
    np.testing.assert_allclose(plot.dimension_values(1), mdata.obsm["X_joint"][:, 1])


@pytest.mark.parametrize("column", ["condition", "score"])
def test_colour_by_obs_column(hv: ModuleType, mdata: MuData, column: str) -> None:
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs[column]])
    assert list(plot.dimension_values(2)) == list(mdata.obs[column])


def test_view_of_a_mudata(hv: ModuleType, mdata: MuData) -> None:
    from hv_anndata import A

    plot = hv.Scatter(mdata[:10], A.obsm["X_joint"][0], [A.obsm["X_joint"][1]])
    assert len(plot) == 10


def test_select(hv: ModuleType, mdata: MuData) -> None:
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs["condition"]])
    chosen = plot.select(**{repr(A.obs["condition"]): "a"})
    assert len(chosen) == int((mdata.obs["condition"] == "a").sum())


def test_renders_with_bokeh(hv: ModuleType, mdata: MuData) -> None:
    from hv_anndata import A

    pytest.importorskip("bokeh")
    hv.extension("bokeh")
    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs["score"]]).opts(
        backend="bokeh", color=A.obs["score"], colorbar=True
    )
    assert len(hv.render(plot, backend="bokeh").renderers) == 1


def test_anndata_still_works(hv: ModuleType, mdata: MuData) -> None:
    from hv_anndata import A

    plot = hv.Scatter(mdata["mod1"], A.X[:, "mod1_g0"], [A.X[:, "mod1_g1"]])
    assert len(plot) == mdata.n_obs


# --- scanpy's HoloViews plotting functions accept a MuData -------------------------------------------


def test_scanpy_scatter(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    from hv_anndata import A

    plot = sc.pl.scatter(mdata, A.obsm["X_joint"][:, [0, 1]], color=A.obs["condition"])
    assert len(plot) == mdata.n_obs


def test_scanpy_umap(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    from hv_anndata import A

    plot = sc.pl.umap(mdata, color=A.obs["score"])
    assert len(plot) == mdata.n_obs


# --- Muon's own plotting functions under the v2 preset ----------------------------------------------


def test_pl_embedding_v2(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.embedding(mdata, basis="mod1:umap", color="condition")
    assert isinstance(plot, hv.Scatter)
    np.testing.assert_allclose(plot.dimension_values(0), mdata.mod["mod1"].obsm["umap"][:, 0])
    assert list(plot.dimension_values(2)) == list(mdata.obs["condition"])


def test_pl_embedding_v2_joint_basis_and_no_colour(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.embedding(mdata, basis="X_joint")
    assert isinstance(plot, hv.Scatter)
    assert len(plot) == mdata.n_obs


def test_pl_embedding_v2_feature_colour(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.embedding(mdata, basis="mod1:umap", color="mod1_g0")
    assert isinstance(plot, hv.Scatter)
    np.testing.assert_allclose(plot.dimension_values(2), mdata.mod["mod1"][:, "mod1_g0"].X.ravel())


def test_pl_embedding_v2_several_colours(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    layout = mu.pl.embedding(mdata, basis="mod1:umap", color=["condition", "mod2_g0"])
    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2


def test_pl_umap_v2(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.umap(mdata, color="score")
    assert isinstance(plot, hv.Scatter)
    assert len(plot) == mdata.n_obs


def test_pl_scatter_v2(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.scatter(mdata, x="mod1_g0", y="mod2_g0", color="condition")
    assert isinstance(plot, hv.Scatter)
    np.testing.assert_allclose(plot.dimension_values(0), mdata.mod["mod1"][:, "mod1_g0"].X.ravel())
    np.testing.assert_allclose(plot.dimension_values(1), mdata.mod["mod2"][:, "mod2_g0"].X.ravel())


def test_pl_v2_options_are_holoviews_options(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    plot = mu.pl.embedding(mdata, basis="mod1:umap", color="condition", size=7)
    assert hv.Store.lookup_options("bokeh", plot, "style").kwargs["size"] == 7


def test_pl_v2_anndata_rejects_features_and_raw(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    import muon as mu

    mod = mdata.mod["mod1"]
    with pytest.raises(NotImplementedError, match="only be coloured by .obs"):
        mu.pl.embedding(mod, basis="umap", color="mod1_g0")
    with pytest.raises(NotImplementedError, match="use_raw"):
        mu.pl.embedding(mod, basis="umap", use_raw=True)


def test_pl_v2_has_a_hover_tool_by_default(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    from bokeh.models import HoverTool

    import muon as mu

    plot = mu.pl.embedding(mdata, basis="mod1:umap", color="condition")
    assert any(isinstance(t, HoverTool) for t in hv.render(plot, backend="bokeh").tools)


def test_pl_v2_tools_can_be_overridden(hv: ModuleType, scanpy_v2: None, mdata: MuData) -> None:
    from bokeh.models import HoverTool

    import muon as mu

    plot = mu.pl.embedding(mdata, basis="mod1:umap", color="condition", tools=["pan"])
    assert not any(isinstance(t, HoverTool) for t in hv.render(plot, backend="bokeh").tools)
