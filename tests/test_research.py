"""Tests for the from-scratch attention model in flowbook.research.

Two different kinds of claims are tested here, deliberately kept separate:

  1. Correctness of the hand-derived backward pass (`grad_check`) -- this
     must hold exactly, always, regardless of the data. It is the load-
     bearing test in this file.
  2. That the training loop actually reduces loss / learns to fit its own
     training data (a basic sanity check that the optimizer and forward/
     backward wiring are hooked up correctly).

What is *not* asserted: that the model beats a majority-class baseline on
held-out, independently-sampled simulation seeds. Measured out-of-sample
accuracy on this synthetic multi-regime data hovers close to the majority
baseline in practice (see docs/RESEARCH.md, "Known limitations") -- a
flaky assertion chasing a specific out-of-sample number would be testing
noise, not code. See scripts/train_research_model.py for a full run that
prints in-sample and out-of-sample numbers transparently instead of
asserting a threshold.
"""

import numpy as np

from flowbook.research import (
    ModelConfig,
    TinyAttentionClassifier,
    build_dataset,
    class_balance,
    evaluate,
    grad_check,
    label_from_return,
    positional_encoding,
    train,
)
from flowbook.simulator import research_sim_config


def test_positional_encoding_shape_and_bounds():
    pe = positional_encoding(seq_len=10, d_model=8)
    assert pe.shape == (10, 8)
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0)  # sin/cos range


def test_label_from_return_thresholds():
    assert label_from_return(0.01, alpha=0.001) == 2   # up
    assert label_from_return(-0.01, alpha=0.001) == 0  # down
    assert label_from_return(0.0, alpha=0.001) == 1     # flat
    assert label_from_return(0.0005, alpha=0.001) == 1  # inside threshold -> flat


def test_grad_check_passes_on_random_input():
    """The core correctness claim: every analytic gradient in
    TinyAttentionClassifier.backward matches a numerical (finite-difference)
    gradient to within tolerance. See module docstring for why this matters
    (this stands in for the autograd framework this project deliberately
    doesn't use).
    """
    model = TinyAttentionClassifier(ModelConfig(feature_dim=6, d_model=8, seq_len=5, seed=2))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 6))
    errors = grad_check(model, X, y=1)
    assert all(err < 2e-2 for err in errors.values())


def test_grad_check_passes_for_each_class_label():
    # Exercise the cross-entropy gradient at all three class targets, since
    # dlogits differs structurally for the correct vs. incorrect classes.
    model = TinyAttentionClassifier(ModelConfig(feature_dim=4, d_model=6, seq_len=4, seed=3))
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 4))
    for label in (0, 1, 2):
        errors = grad_check(model, X, y=label)
        assert all(err < 2e-2 for err in errors.values())


def test_build_dataset_shapes_and_label_range():
    X, y = build_dataset(n_sequences=40, seq_len=10, horizon=50,
                          sim_config=research_sim_config(seed=5))
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] <= 40
    assert X.shape[1] == 10
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_training_reduces_loss_and_fits_training_data_above_chance():
    """Sanity check for the training loop itself: after a few epochs the
    model should fit its own (small) training set well above the 1/3
    chance level for a 3-class problem. This is a test of the optimizer/
    forward/backward wiring, not a claim about generalization.
    """
    X, y = build_dataset(n_sequences=120, seq_len=12, horizon=100,
                          sim_config=research_sim_config(seed=11))
    assert len(np.unique(y)) >= 2, "degenerate dataset (single class) -- test is not meaningful"

    model = TinyAttentionClassifier(ModelConfig(feature_dim=X.shape[2], seq_len=X.shape[1], d_model=8, seed=0))
    history = train(model, X, y, epochs=8, lr=5e-3, verbose=False)

    assert history[-1] < history[0], "training loss did not decrease"
    train_acc = evaluate(model, X, y)
    assert train_acc > 0.4, f"expected the model to fit its own training data reasonably well, got {train_acc}"


def test_class_balance_helper():
    y = np.array([0, 0, 1, 2, 2, 2])
    balance = class_balance(y)
    assert balance == {"down": 2, "flat": 1, "up": 3}
