"""End-to-end training + honest evaluation run for the from-scratch
attention model in flowbook.research.

Run with:  python scripts/train_research_model.py

This intentionally trains on many independent simulation seeds and
evaluates on a disjoint set of independent seeds (rather than a chronological
split of one simulation run), because the synthetic order-flow-imbalance
regime is autocorrelated: a chronological split's "test" period is often
just the tail of the same regime episode the model trained on, which
inflates apparent accuracy. Splitting by seed instead means every held-out
example comes from an independently sampled regime path.

Prints in-sample and out-of-sample accuracy plus the corresponding
majority-class baselines, without asserting either -- see
docs/RESEARCH.md, "Known limitations" for a discussion of what the
out-of-sample numbers here actually mean (and don't).
"""

from __future__ import annotations

import numpy as np

from flowbook.research import ModelConfig, TinyAttentionClassifier, build_dataset, class_balance, evaluate, train
from flowbook.simulator import research_sim_config

TRAIN_SEEDS = range(1, 31)
TEST_SEEDS = range(1000, 1015)
SEQ_LEN = 20
HORIZON = 150
N_PER_SEED = 150


def build_multi_seed_dataset(seeds) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seed in seeds:
        X, y = build_dataset(
            n_sequences=N_PER_SEED, seq_len=SEQ_LEN, horizon=HORIZON,
            sim_config=research_sim_config(seed=seed),
        )
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    print(f"Building training set from {len(list(TRAIN_SEEDS))} independent simulation seeds...")
    X_train, y_train = build_multi_seed_dataset(TRAIN_SEEDS)
    print(f"  {len(X_train)} sequences, class balance {class_balance(y_train)}")

    print(f"Building held-out test set from {len(list(TEST_SEEDS))} *disjoint* seeds...")
    X_test, y_test = build_multi_seed_dataset(TEST_SEEDS)
    print(f"  {len(X_test)} sequences, class balance {class_balance(y_test)}")

    model = TinyAttentionClassifier(
        ModelConfig(feature_dim=X_train.shape[2], seq_len=X_train.shape[1], d_model=16, seed=0)
    )

    print("\nTraining...")
    train(model, X_train, y_train, epochs=15, lr=3e-3, verbose=True)

    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)
    train_baseline = max(class_balance(y_train).values()) / len(y_train)
    test_baseline = max(class_balance(y_test).values()) / len(y_test)

    print("\n" + "=" * 60)
    print(f"in-sample  accuracy: {train_acc:.3f}   (majority-class baseline: {train_baseline:.3f})")
    print(f"out-of-sample accuracy: {test_acc:.3f}   (majority-class baseline: {test_baseline:.3f})")
    print("=" * 60)
    print(
        "\nSee docs/RESEARCH.md ('Known limitations') for why the "
        "out-of-sample number should be read as a proof-of-pipeline result, "
        "not a claim of predictive edge on real markets."
    )


if __name__ == "__main__":
    main()
