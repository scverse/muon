"""Tests for the HoloViews interface for MuData (``muon._hv``)."""

from types import SimpleNamespace

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
        mods[name] = mod
    mdata = MuData(mods)
    mdata.obs["condition"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    mdata.obs["score"] = rng.normal(size=n)
    mdata.obsm["X_joint"] = rng.normal(size=(n, 2))
    mdata.obsm["umap"] = rng.normal(size=(n, 2))
    return mdata


@pytest.fixture
def hv():
    """HoloViews, with the MuData interface registered."""
    hv = pytest.importorskip("holoviews")
    hv_anndata = pytest.importorskip("hv_anndata")
    hv_anndata.register()  # what `sc.pl.hv_init()` does for AnnData
    _hv.register_mudata_interface()
    return hv


@pytest.fixture
def scanpy_v2():
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


# --- when is the interface registered? ---------------------------------------------------------------


def test_default_preset_is_not_v2():
    assert not _hv._scanpy_preset_is_v2()


@pytest.mark.skipif(not hasattr(sc, "Preset"), reason="this scanpy version has no presets")
def test_preset_is_v2(monkeypatch):
    monkeypatch.setattr(sc, "settings", SimpleNamespace(preset=sc.Preset.ScanpyV2Preview))
    assert _hv._scanpy_preset_is_v2()


def test_scanpy_without_presets(monkeypatch):
    monkeypatch.delattr(sc, "Preset", raising=False)
    assert not _hv._scanpy_preset_is_v2()


@pytest.mark.parametrize("is_v2", [False, True])
def test_register_only_if_v2(monkeypatch, is_v2):
    calls = []
    monkeypatch.setattr(_hv, "_scanpy_preset_is_v2", lambda: is_v2)
    monkeypatch.setattr(_hv, "register_mudata_interface", lambda: calls.append(1))
    assert _hv.register_if_scanpy_v2() is is_v2
    assert calls == ([1] if is_v2 else [])


def test_missing_holoviews_warns_instead_of_failing(monkeypatch):
    def fail():
        raise ImportError("no holoviews")

    monkeypatch.setattr(_hv, "_scanpy_preset_is_v2", lambda: True)
    monkeypatch.setattr(_hv, "register_mudata_interface", fail)
    with pytest.warns(UserWarning, match="no holoviews"):
        assert _hv.register_if_scanpy_v2() is False


# --- the interface itself ---------------------------------------------------------------------------


def test_registration(hv):
    assert "mudata" in hv.core.data.datatypes
    assert "mudata" in hv.core.Interface.interfaces
    _hv.register_mudata_interface()  # a second call does nothing
    assert hv.core.data.datatypes.count("mudata") == 1


def test_scatter(hv, mdata):
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1]])
    assert len(plot) == mdata.n_obs
    np.testing.assert_allclose(plot.dimension_values(0), mdata.obsm["X_joint"][:, 0])
    np.testing.assert_allclose(plot.dimension_values(1), mdata.obsm["X_joint"][:, 1])


@pytest.mark.parametrize("column", ["condition", "score"])
def test_colour_by_obs_column(hv, mdata, column):
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs[column]])
    assert list(plot.dimension_values(2)) == list(mdata.obs[column])


def test_view_of_a_mudata(hv, mdata):
    from hv_anndata import A

    plot = hv.Scatter(mdata[:10], A.obsm["X_joint"][0], [A.obsm["X_joint"][1]])
    assert len(plot) == 10


def test_select(hv, mdata):
    from hv_anndata import A

    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs["condition"]])
    chosen = plot.select(**{repr(A.obs["condition"]): "a"})
    assert len(chosen) == int((mdata.obs["condition"] == "a").sum())


def test_renders_with_bokeh(hv, mdata):
    from hv_anndata import A

    pytest.importorskip("bokeh")
    hv.extension("bokeh")
    plot = hv.Scatter(mdata, A.obsm["X_joint"][0], [A.obsm["X_joint"][1], A.obs["score"]]).opts(
        backend="bokeh", color=A.obs["score"], colorbar=True
    )
    assert len(hv.render(plot, backend="bokeh").renderers) == 1


def test_anndata_still_works(hv, mdata):
    from hv_anndata import A

    plot = hv.Scatter(mdata["mod1"], A.X[:, "mod1_g0"], [A.X[:, "mod1_g1"]])
    assert len(plot) == mdata.n_obs


# --- scanpy's HoloViews plotting functions accept a MuData -------------------------------------------


def test_scanpy_scatter(hv, scanpy_v2, mdata):
    from hv_anndata import A

    plot = sc.pl.scatter(mdata, A.obsm["X_joint"][:, [0, 1]], color=A.obs["condition"])
    assert len(plot) == mdata.n_obs


def test_scanpy_umap(hv, scanpy_v2, mdata):
    from hv_anndata import A

    plot = sc.pl.umap(mdata, color=A.obs["score"])
    assert len(plot) == mdata.n_obs
