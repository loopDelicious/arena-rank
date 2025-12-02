"""Contextual Bradley-Terry rating system implementation in JAX."""

from functools import partial
import math
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import PyTree

from arena.models.rating_system import RatingSystem
from arena.utils.data_utils import ContextualPairDataset
from arena.models.bradley_terry import accumulate_matrix

jax.config.update("jax_enable_x64", True)


@jit
def loss_function(params: PyTree, data: PyTree, reg: float, clip=1e-15) -> jnp.ndarray:
    """
    Computes the Contextual Bradley-Terry loss.
    """
    ratings = params["ratings"]
    coeffs = params["coeffs"]
    matchups = data["pairs"]
    features = data["features"]
    outcomes = data["outcomes"]
    weights = data["weights"]

    matchup_ratings = ratings[matchups]
    bt_logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    context_logits = jnp.dot(features, coeffs)
    total_logits = bt_logits + context_logits
    probs = jax.nn.sigmoid(total_logits)
    probs_clipped = jnp.clip(probs, clip, 1.0 - clip)
    log_likelihood = outcomes * jnp.log(probs_clipped) + (1.0 - outcomes) * jnp.log(1.0 - probs_clipped)

    weighted_ll = jnp.mean(weights * log_likelihood)
    reg_loss = 0.5 * reg * jnp.sum(coeffs**2)
    return -(weighted_ll) + reg_loss


@partial(jit, static_argnames=["n_obs", "n_competitors", "n_features"])
def compute_hessian_and_covvariance(
    ratings: jnp.ndarray,
    coeffs: jnp.ndarray,
    matchups: jnp.ndarray,
    features: jnp.ndarray,
    outcomes: jnp.ndarray,
    weights: jnp.ndarray,
    reg: float,
    hessian_reg: float,
    n_competitors: int,
    n_features: int,
    n_obs: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the Hessian and Gradient Covariance matrices for the
    Sandwich Estimator within the Contextual BT framework.

    Constructs the block matrix:
    [ hessian_ratings   hessian_cross ]
    [ hessian_cross.T   hessian_feats ]
    """
    matchup_ratings = ratings[matchups]
    bt_logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    context_logits = jnp.dot(features, coeffs)
    probs = jax.nn.sigmoid(bt_logits + context_logits)

    # row level second derivative values
    hess_vals = (probs * (1.0 - probs) * weights) / n_obs

    idx_a = matchups[:, 0]
    idx_b = matchups[:, 1]

    # rating block of the hessian
    hess_ratings = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
    hess_ratings = accumulate_matrix(hess_ratings, hess_vals, idx_a, idx_b)
    hess_ratings = hess_ratings + jnp.eye(n_competitors) * hessian_reg

    # feature block of the hessian
    hess_features = jnp.dot(features.T, (hess_vals[:, None] * features))
    hess_features = hess_features + jnp.eye(n_features) * hessian_reg

    # "cross" block of the hessian, deriv w.r.t. ratings and features
    hess_weighted_feats = hess_vals[:, None] * features  # (n_obs, n_features)
    hess_cross = jnp.zeros((n_competitors, n_features), dtype=ratings.dtype)
    hess_cross = hess_cross.at[idx_a, :].add(hess_weighted_feats)
    hess_cross = hess_cross.at[idx_b, :].add(-hess_weighted_feats)

    hessian = jnp.block(
        [
            [hess_ratings, hess_cross],
            [hess_cross.T, hess_features],
        ]
    )

    # row level gradient covariance values
    grad_cov_vals = (((outcomes - probs) * weights) ** 2) / n_obs

    # rating block of the gradient covariance
    grad_cov_ratings = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
    grad_cov_ratings = accumulate_matrix(grad_cov_ratings, grad_cov_vals, idx_a, idx_b)

    # feature block of the gradient covariance
    grad_cov_features = jnp.dot(features.T, (grad_cov_vals[:, None] * features))

    # cross block of the gradient covariance
    grad_cov_weighted_feats = grad_cov_vals[:, None] * features  # (n_obs, n_features)
    grad_cov_cross = jnp.zeros((n_competitors, n_features), dtype=ratings.dtype)
    grad_cov_cross = grad_cov_cross.at[idx_a, :].add(grad_cov_weighted_feats)
    grad_cov_cross = grad_cov_cross.at[idx_b, :].add(-grad_cov_weighted_feats)

    grad_cov = jnp.block(
        [
            [grad_cov_ratings, grad_cov_cross],
            [grad_cov_cross.T, grad_cov_features],
        ]
    )

    # correct for L2 regularization in gradient covariance
    params_vec = jnp.concatenate([jnp.zeros(n_competitors), coeffs])
    reg_correction = (reg**2 / n_obs) * jnp.outer(params_vec, params_vec)
    grad_cov = grad_cov - reg_correction
    return hessian, grad_cov


class ContextualBradleyTerry(RatingSystem):
    """
    Bradley-Terry rating system with contextual features.
    """

    def __init__(
        self,
        n_competitors: int,
        n_features: int,
        reg: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        dtype=jnp.float64,
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        hessian_reg: float = 1e-5,
        clip: float = 1e-15,
    ):
        self.n_competitors = n_competitors
        self.n_features = n_features
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.clip = clip
        self.dtype = dtype
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.hessian_reg = hessian_reg
        self.fitted = False
        self.alpha = scale / math.log(base)
        self.params = {
            "ratings": jnp.zeros(n_competitors, dtype=dtype),
            "coeffs": jnp.zeros(n_features, dtype=dtype),
        }

    def fit(self, dataset: ContextualPairDataset):
        """
        Fit the Contextual Bradley-Terry model.

        Args:
            dataset: ContextualPairDataset containing matchups, outcomes, features.
        """
        features = dataset.features
        initial_params = self.params
        data = {
            "pairs": dataset.pairs,
            "weights": dataset.weights,
            "outcomes": dataset.outcomes,
            "features": features,
        }

        loss_fn = partial(loss_function, reg=self.reg, data=data, clip=self.clip)
        optimized_params, _ = self.lbfgs_minimize(
            loss_function=loss_fn,
            initial_params=initial_params,
            max_iter=self.max_iter,
            gtol=self.tol,
            ftol=self.tol,
        )
        self.params = optimized_params
        self.fitted = True
        return self

    def compute_ratings_and_cis(
        self, dataset: ContextualPairDataset, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculates ratings, coefficients, and CLT-based confidence intervals.
        """
        if not self.fitted:
            self.fit(dataset)

        features = dataset.features
        ratings = self.params["ratings"]
        coeffs = self.params["coeffs"]
        n_obs = dataset.pairs.shape[0]

        hessian, gradient_cov = compute_hessian_and_covvariance(
            ratings,
            coeffs,
            dataset.pairs,
            features,
            dataset.outcomes,
            dataset.weights,
            self.reg,
            self.hessian_reg,
            self.n_competitors,
            self.n_features,
            n_obs,
        )
        hessian_inv = jnp.linalg.inv(hessian)
        asymptotic_variance = hessian_inv @ gradient_cov @ hessian_inv

        param_variances = jnp.diag(asymptotic_variance) / n_obs
        std_errs = jnp.sqrt(param_variances)

        rating_variances = param_variances[: self.n_competitors]
        rating_std_errs = std_errs[: self.n_competitors]
        z_score = jax.scipy.stats.norm.ppf(1 - significance_level / 2)
        interval_widths = z_score * rating_std_errs

        offset = self.init_rating
        scaled_ratings = ratings * self.alpha + offset
        scaled_widths = interval_widths * self.alpha
        scaled_variances = rating_variances * (self.alpha**2)

        return {
            "competitors": dataset.competitors,
            "ratings": scaled_ratings,
            "coeffs": coeffs,
            "rating_lower": scaled_ratings - scaled_widths,
            "rating_upper": scaled_ratings + scaled_widths,
            "variances": scaled_variances,
        }
