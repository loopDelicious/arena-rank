import random
from itertools import combinations
import numpy as np
import pandas as pd
from arena.utils.data_utils import get_matchups_and_competitors, PairDataset


def test_get_matchups_and_competitors() -> None:
    col_a = ["D", "A", "B", "C"]
    col_b = ["B", "C", "D", "A"]
    df = pd.DataFrame({"col_a": col_a, "col_b": col_b})
    matchups, competitors = get_matchups_and_competitors(df, competator_cols=["col_a", "col_b"])
    assert competitors == ["A", "B", "C", "D"]
    expected_matchups = np.array([[3, 1], [0, 2], [1, 3], [2, 0]])
    np.testing.assert_array_equal(matchups, expected_matchups)


def test_pair_dataset():
    random.seed(42)
    n_competitors = 4
    competitors = [f"competitor_{i}" for i in range(n_competitors)]
    competitor_to_idx = {name: i for i, name in enumerate(competitors)}
    combs = list(combinations(competitors, 2))
    comb_counts = [100 * (i + 1) for i in range(len(combs))]
    comb_winrates = [random.uniform(0.1, 0.9) for _ in range(len(combs))]
    num_pairs = sum(comb_counts)
    matches = []
    win_counts = np.zeros((n_competitors, n_competitors))
    for (a, b), count, winrate in zip(combs, comb_counts, comb_winrates):
        a_wins = int(count * winrate)
        b_wins = count - a_wins
        win_counts[competitor_to_idx[a], competitor_to_idx[b]] = a_wins
        win_counts[competitor_to_idx[b], competitor_to_idx[a]] = b_wins

        matches.extend([(a, b, "model_a")] * a_wins)
        matches.extend([(b, a, "model_b")] * b_wins)
    random.shuffle(matches)
    df = pd.DataFrame(matches, columns=["model_a", "model_b", "winner"])
    dataset = PairDataset.from_pandas(
        df,
        competitor_cols=["model_a", "model_b"],
        outcome_col="winner",
        outcome_map={"model_a": 1.0, "model_b": 0.0},
    )
    assert dataset.n_competitors == n_competitors
    assert dataset.n_pairs == len(combs) * 2  # each combination has two outcomes
    total_counts = np.sum(dataset.counts)
    assert total_counts == num_pairs
    wins = dataset.outcomes == 1.0
    for (i, j), count in zip(dataset.pairs[wins], dataset.counts[wins]):
        a_wins = win_counts[i, j]
        assert count == a_wins
    for (j, i), count in zip(dataset.pairs[~wins], dataset.counts[~wins]):
        b_wins = win_counts[j, i]
        assert count == b_wins
