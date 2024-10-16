import muon as mu
from anndata.tests.helpers import assert_equal


class TestFilter:
    def test_filter_obs_simple(self, pbmc3k_processed):
        A = pbmc3k_processed[:500,].copy()
        A_subset = A[A.obs["louvain"] == "B cells"].copy()
        B = pbmc3k_processed[500:,].copy()
        B_subset = B[B.obs["louvain"] == "NOT HERE"].copy()
        mdata = mu.MuData({"A": A, "B": B})
        mu.pp.filter_obs(mdata, "A:louvain", lambda x: x == "B cells")
        assert mdata["B"].n_obs == 0
        assert mdata["A"].obs["louvain"].unique() == "B cells"
        assert B.n_obs == 0
        assert A.obs["louvain"].unique() == "B cells"
        assert_equal(mdata["A"], A_subset)
        assert_equal(mdata["B"], B_subset)
