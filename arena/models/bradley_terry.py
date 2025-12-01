from functools import partial
import math
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import PyTree

from arena.models.rating_system import RatingSystem
from arena.utils.data_utils import PairDataset

jax.config.update("jax_enable_x64", True)


@jit
def accumulate_matrix(mat, vals, idx_a, idx_b):
    mat = mat.at[idx_a, idx_a].add(vals)
    mat = mat.at[idx_b, idx_b].add(vals)
    mat = mat.at[idx_a, idx_b].add(-vals)
    mat = mat.at[idx_b, idx_a].add(-vals)
    return mat


@jit
def loss_function(params, data: PyTree) -> jnp.ndarray:
    ratings = params["ratings"]
    matchups = data["pairs"]
    weights = data["weights"]
    outcomes = data["outcomes"]
    rating_diffs = ratings[matchups[:, 0]] - ratings[matchups[:, 1]]
    probs = jax.nn.sigmoid(rating_diffs)
    loss = -jnp.sum(
        weights * (outcomes * jnp.log(probs + 1e-15) + (1 - outcomes) * jnp.log(1 - probs + 1e-15))
    ) / jnp.sum(weights)
    return loss


@partial(jit, static_argnames=["n_competitors"])
def _compute_clt_stats(
    ratings: jnp.ndarray,
    matchups: jnp.ndarray,
    outcomes: jnp.ndarray,
    counts: jnp.ndarray,
    opt_weights: jnp.ndarray,
    hessian_reg: float,
    n_competitors: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the Hessian and Gradient Covariance matrices for the sandwich estimator.

    Args:
        ratings: Competitor ratings, shape (n_competitors,)
        matchups: Matchup pairs, shape (n_pairs, 2)
        outcomes: Outcomes of matchups, shape (n_pairs,)
        counts: Counts of each unique pair, shape (n_pairs,)
        opt_weights: Optimization weights for each matchup, incorporates both occurance counts and reweighting, shape (n_pairs,)
        hessian_reg: Regularization term for Hessian diagonal.
        n_competitors: Total number of competitors.
    """
    matchup_ratings = ratings[matchups]
    logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
    probs = jax.nn.sigmoid(logits)

    grad_vals = (outcomes - probs) * opt_weights
    # variance per unique pair bucket
    var_grad_vals = (grad_vals**2) / counts
    hess_vals = probs * (1.0 - probs) * opt_weights

    idx_a = matchups[:, 0]
    idx_b = matchups[:, 1]

    hessian = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
    grad_cov = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)

    hessian = accumulate_matrix(hessian, hess_vals, idx_a, idx_b)
    grad_cov = accumulate_matrix(grad_cov, var_grad_vals, idx_a, idx_b)

    # regularize the diagonal of the Hessian for numerical stability
    hessian = hessian + (jnp.eye(n_competitors) * hessian_reg)
    return hessian, grad_cov


class BradleyTerry(RatingSystem):
    """Bradley-Terry rating system implementation."""

    def __init__(
        self,
        n_competitors,
        max_iter=1000,
        tol=1e-6,
        dtype=jnp.float64,
        # Scaling/output parameters
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        hessian_reg: float = 1e-5,
    ):
        self.n_competitors = n_competitors
        self.max_iter = max_iter
        self.tol = tol
        self.dtype = dtype
        self.params = {"ratings": jnp.zeros(n_competitors, dtype=dtype)}
        self.fitted = False

        # Formatting/Scaling configs
        self.scale = scale
        self.base = base
        self.alpha = scale / math.log(base)
        self.init_rating = init_rating
        self.hessian_reg = hessian_reg

    def fit(self, dataset: PairDataset):
        """Fit the Bradley-Terry model to the provided dataset."""
        initial_params = self.params
        data = {
            "pairs": dataset.pairs,
            "weights": dataset.opt_weights,
            "outcomes": dataset.outcomes,
        }
        optimized_params, final_state = self.lbfgs_minimize(
            loss_function=partial(loss_function, data=data),
            initial_params=initial_params,
            max_iter=self.max_iter,
            gtol=self.tol,
            ftol=self.tol,
        )
        self.params = optimized_params
        self.fitted = True
        return self

    def compute_ratings_and_cis(self, dataset: PairDataset, significance_level: float = 0.05):
        """
        Fits the model (if needed), calculates CLT-based confidence intervals
        """
        # 1. Fit if not already fitted
        if not self.fitted:
            self.fit(dataset)

        ratings = self.params["ratings"]
        total_battles = jnp.sum(dataset.counts)

        is_unweighted = jnp.allclose(dataset.weights, 1.0)
        reg_factor = jnp.where(is_unweighted, total_battles, 1.0)
        effective_reg = self.hessian_reg * reg_factor

        # 2. Compute Statistics with effective regularization
        hessian, gradient_cov = _compute_clt_stats(
            ratings,
            dataset.pairs,
            dataset.outcomes,
            dataset.counts,
            dataset.opt_weights,
            effective_reg,
            self.n_competitors,
        )

        hessian_inv = jnp.linalg.inv(hessian)

        # This is the full Covariance Matrix (Sigma), not the variance vector
        covariance_matrix = hessian_inv @ gradient_cov @ hessian_inv

        # This is the actual variance vector
        variance = jnp.diag(covariance_matrix)

        std_errs = jnp.sqrt(variance)
        z_score = jax.scipy.stats.norm.ppf(1 - significance_level / 2)
        interval_widths = z_score * std_errs

        offset = self.init_rating
        scaled_ratings = ratings * self.alpha + offset
        scaled_widths = interval_widths * self.alpha
        scaled_variance = variance * (self.alpha**2)

        print(f"{scaled_ratings.shape=}")
        print(f"{scaled_widths.shape=}")
        print(f"{scaled_variance.shape=}")

        return {
            "models": dataset.competitors,
            "ratings": scaled_ratings,
            "rating_lower": scaled_ratings - scaled_widths,
            "rating_upper": scaled_ratings + scaled_widths,
            "variance": scaled_variance,
        }
