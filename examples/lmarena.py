import datasets
import jax.numpy as jnp
from arena.utils.data_utils import ContextualPairDataset
from arena.models.contextual_bradley_terry import ContextualBradleyTerry


def main():
    dataset = datasets.load_dataset(
        "lmarena-ai/arena-human-preference-140k",
        columns=["model_a", "model_b", "winner", "conv_metadata"],
        split="train",
    )
    df = dataset.to_pandas()
    df["sum_assistant_tokens_a"] = df["conv_metadata"].apply(lambda x: x["sum_assistant_a_tokens"])
    df["sum_assistant_tokens_b"] = df["conv_metadata"].apply(lambda x: x["sum_assistant_b_tokens"])
    df["header_count_a"] = df["conv_metadata"].apply(lambda x: sum(x["header_count_a"].values()))
    df["header_count_b"] = df["conv_metadata"].apply(lambda x: sum(x["header_count_b"].values()))
    df["bold_count_a"] = df["conv_metadata"].apply(lambda x: sum(x["bold_count_a"].values()))
    df["bold_count_b"] = df["conv_metadata"].apply(lambda x: sum(x["bold_count_b"].values()))
    df["list_count_a"] = df["conv_metadata"].apply(lambda x: sum(x["list_count_a"].values()))
    df["list_count_b"] = df["conv_metadata"].apply(lambda x: sum(x["list_count_b"].values()))

    feature_cols = ["sum_assistant_tokens", "header_count", "bold_count", "list_count"]
    for feature in feature_cols:
        diff = df[f"{feature}_a"] - df[f"{feature}_b"]
        sum_ = df[f"{feature}_a"] + df[f"{feature}_b"]
        sum_[sum_ == 0] = 1  # prevent division by zero
        df[feature] = diff / sum_
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

    dataset = ContextualPairDataset.from_pandas(
        df,
        competitor_cols=["model_a", "model_b"],
        outcome_col="winner",
        outcome_map={"model_a": 1.0, "model_b": 0.0, "tie": 0.5, "both_bad": 0.5},
        feature_cols=feature_cols,
        reweighted=True,
        min_pair_count=100,
    )
    model = ContextualBradleyTerry(
        n_competitors=len(dataset.competitors),
        n_features=len(feature_cols),
    )
    results = model.compute_ratings_and_cis(dataset, significance_level=0.05)
    ratings = results["ratings"]
    coeffs = results["coeffs"]

    sorted_indices = jnp.argsort(-ratings)
    for idx in range(len(dataset.competitors)):
        competitor_idx = sorted_indices[idx]
        competitor_name = dataset.competitors[competitor_idx]
        rating = ratings[competitor_idx]
        print(f"{idx + 1:3d}. {competitor_name:30s} Rating: {rating:.3f}")
    print("\nFeature Coefficients:")
    for i, coef in enumerate(coeffs):
        print(f"  {feature_cols[i]:30s} Coefficient: {coef:.3f}")


if __name__ == "__main__":
    main()
