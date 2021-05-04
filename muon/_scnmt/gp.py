import numpy as np
from anndata import AnnData
from .._core.mudata import MuData
from .._scnmt.utils import GenomeRegion

from typing import List, Union, Optional, Callable, Iterable
from . import utils
from . import tools as tl
from collections import OrderedDict
import gpflow


def fetch_region_gp(data: Union[AnnData, MuData], region: Union[str, GenomeRegion]):
    """
    Fetch region from tabix file and return in data frame
    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'met' modality.
    region
        Genomic Region to fetch. Can be specified as region string or as GenomeRegion object.
    return_region
        Whether to return a DataFrame with the methylation events or just store it in data.uns['active_region']
    """
    adata = utils.get_modality(data, modality="met")

    region_df = tl.fetch_region(adata, region)
    data = adata.obs.merge(region_df, on="id_met")
    X = data[["pseudotime", "pos"]].values
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (0, 1))
    X[:, 1] = utils.center_colum(X[:, 1])
    y = data[["rate"]].values >= 0.5
    y = y.astype(np.int)
    return X, y


def fit_model(X, Y, likelihood, kernel, n_ind=5):
    import gpflow

    # import robustgp

    # X, Y = fetch_region_gp(adata, region)
    Z = utils.make_grid_along_input(X, num_points=n_ind)  # num points per dimension
    # M = 100  # We choose 1000 inducing variables
    # init_method = robustgp.ConditionalVariance()
    # Z = init_method.compute_initialisation(X[:, :2], M, kernel)[0]
    print(Z)
    meanf = get_meanf(Y)

    model = gpflow.models.SVGP(
        kernel=kernel, likelihood=likelihood, inducing_variable=Z  # , mean_function=meanf
    )
    # gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    train(model, X=X, y=Y)
    return model


def fit_model_VGP(X, Y, likelihood, kernel, n_ind=5):
    import gpflow

    # X, Y = fetch_region_gp(adata, region)
    # Z = utils.make_grid_along_input(X, num_points=n_ind)  # num points per dimension
    meanf = get_meanf(Y)

    m = gpflow.models.VGP(data=(X, Y), kernel=kernel, likelihood=likelihood, mean_function=meanf)
    # gpflow.utilities.set_trainable(m.inducing_variable.Z, False)
    train(m)
    return m


def train(model, **kwargs):
    import gpflow

    o = gpflow.optimizers.Scipy()
    # @tf.function(autograph=False)
    if hasattr(model, "data"):
        training_loss = model.training_loss_closure()
    else:
        X = kwargs.get("X", None)
        y = kwargs.get("y", None)
        training_loss = model.training_loss_closure((X, y), compile=False)
    o.minimize(training_loss, variables=model.trainable_variables)


def probit(x):
    import tensorflow as tf

    return np.sqrt(2.0) * tf.math.erfinv(2 * x - 1)


def get_meanf(y):
    import gpflow

    avg = y.mean()

    if avg >= 1:
        avg = 0.99999
    elif avg <= 0:
        avg = 0.00001

    meanf = gpflow.mean_functions.Constant(probit(avg))
    return meanf


def loglik_null_ber(y):
    y = y[:, 0]
    pmle = np.mean(y)
    nmet = np.sum(y == 1)
    nunmet = y.shape[0] - nmet
    loglik_null = np.log(pmle) * nmet + np.log(1 - pmle) * nunmet
    return loglik_null


def loglik_null_ber_cat(y, categories):
    liks = []
    for cat in np.unique(categories):
        ys = y[categories == cat]
        pmle = np.mean(ys)
        # print(pmle)
        nmet = np.sum(ys == 1)
        nunmet = ys.shape[0] - nmet
        loglik_null = np.log(pmle) * nmet + np.log(1 - pmle) * nunmet
        liks.append(loglik_null)
    return np.sum(liks)


def get_model_stats(model, X, y, prefix=""):
    import gpflow

    d = gpflow.utilities.utilities.read_values(model)

    d = OrderedDict({prefix + k.replace(".", "_"): v for k, v in d.items() if v.size == 1})
    if hasattr(model, "data"):
        d[prefix + "_log_likelihood"] = model.elbo().numpy()
    else:
        d[prefix + "_log_likelihood"] = model.elbo((X, y)).numpy()

    return d


import tensorflow as tf


class Block(gpflow.kernels.Kernel):
    def __init__(self, n_classes, active_dims=None):
        super().__init__(active_dims)
        self.variances = gpflow.Parameter(
            np.repeat(1.0, n_classes), transform=gpflow.utilities.positive()
        )

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        means = tf.gather(self.variances, tf.cast(X, tf.int32))
        return means * tf.cast(
            tf.equal(X, tf.transpose(X2)), tf.float64
        )  # this returns a 2D tensor

    def K_diag(self, X):
        return tf.reshape(
            tf.gather(self.variances, tf.cast(X, tf.int32)), (-1,)
        )  # this returns a 1D tensor
