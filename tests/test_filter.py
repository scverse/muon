import scanpy as sc
import muon as mu

class TestFilter():
    def test_filter_obs_simple(self):
        adata = sc.datasets.pbmc3k_processed()
        mdata = mu.MuData({
            "A": adata[:500, ].copy(),
            "B": adata[500:, ].copy()
        })
        mu.pp.filter_obs(mdata, "A:louvain", lambda x: x == "B cells")
        assert mdata["B"].n_obs == 0
        assert mdata["A"].obs["louvain"].unique() == "B cells"