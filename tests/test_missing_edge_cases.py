from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from lgbm_boost import FastBooster


@pytest.mark.parametrize(
    ("use_missing", "zero_as_missing"),
    [
        (True, False),   # MissingType::NaN
        (True, True),    # MissingType::Zero
        (False, False),  # MissingType::None
    ],
)
@pytest.mark.parametrize("num_leaves", [15, 31, 50, 95])
def test_missing_modes_and_special_floats(
    use_missing: bool, zero_as_missing: bool, num_leaves: int
) -> None:
    rng = np.random.default_rng(777)
    n_train, n_cols = 1200, 10
    X_train = rng.normal(size=(n_train, n_cols)).astype(np.float64)
    if use_missing:
        X_train[rng.random(X_train.shape) < 0.15] = np.nan
    X_train[rng.random(X_train.shape) < 0.15] = 0.0
    y_train = (X_train[:, 0] > 0).astype(np.int32) ^ (rng.random(n_train) < 0.2)

    booster = lgb.train(
        {
            "objective": "binary",
            "num_leaves": num_leaves,
            "min_data_in_leaf": 3,
            "use_missing": use_missing,
            "zero_as_missing": zero_as_missing,
            "verbose": -1,
        },
        lgb.Dataset(X_train, y_train),
        num_boost_round=60,
    )
    fb = FastBooster(booster)

    # Construct adversarial test matrix with special IEEE-754 float64 values
    X_test = rng.normal(size=(513, n_cols)).astype(np.float64)
    specials = np.array(
        [
            np.nan,
            0.0,
            -0.0,
            1e-36,    # <= 1e-35 LightGBM kZeroThreshold -> treated as Zero!
            -1e-36,
            1e-34,    # > 1e-35 -> NOT treated as Zero!
            -1e-34,
            np.inf,
            -np.inf,
            np.finfo(np.float64).max,
            np.finfo(np.float64).min,
        ],
        dtype=np.float64,
    )
    for idx, val in enumerate(specials):
        X_test[idx * 7 : (idx + 1) * 7, idx % n_cols] = val
    # Also add an all-NaN row and an all-Zero row
    X_test[500, :] = np.nan
    X_test[501, :] = 0.0
    X_test[502, :] = -0.0

    ref = booster.predict(X_test)
    for kernel in ("quickscorer", "compact"):
        pred = fb.predict(X_test, kernel=kernel, num_threads=1)
        assert pred.dtype == np.float64
        np.testing.assert_allclose(pred, ref, rtol=1e-12, atol=1e-12)


def test_exact_float64_threshold_boundary_no_split_flip() -> None:
    """Verify exact float64 thresholds preserve 0 split flips even at +/- 1 float64 ULP."""
    rng = np.random.default_rng(101)
    X_train = rng.uniform(-5.0, 5.0, size=(800, 6)).astype(np.float64)
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]

    booster = lgb.train(
        {"objective": "regression", "num_leaves": 31, "min_data_in_leaf": 3, "verbose": -1},
        lgb.Dataset(X_train, y_train),
        num_boost_round=25,
    )

    dump = booster.dump_model()
    thresholds_64: list[float] = []

    def collect_thresholds(node: dict) -> None:
        if "threshold" in node:
            thresholds_64.append(float(node["threshold"]))
            collect_thresholds(node["left_child"])
            collect_thresholds(node["right_child"])

    for tree in dump["tree_info"]:
        collect_thresholds(tree["tree_structure"])

    assert len(thresholds_64) > 0

    # Build batch hitting exact float64 thresholds +/- 1 float64 ULP (~1e-16)
    rows = []
    for t64 in thresholds_64[:80]:
        v64 = np.float64(t64)
        rows.append(np.full(6, v64, dtype=np.float64))
        rows.append(np.full(6, np.nextafter(v64, np.float64(-np.inf)), dtype=np.float64))
        rows.append(np.full(6, np.nextafter(v64, np.float64(np.inf)), dtype=np.float64))
    X_edge = np.vstack(rows)

    fb = FastBooster(booster)
    ref = booster.predict(X_edge)
    np.testing.assert_allclose(fb.predict(X_edge, kernel="quickscorer"), ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(fb.predict(X_edge, kernel="compact"), ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n_rows", [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 127, 128, 129, 255, 256, 257, 401])
def test_awkward_batch_tail_sizes(n_rows: int) -> None:
    rng = np.random.default_rng(55)
    X_train = rng.normal(size=(400, 8)).astype(np.float64)
    y_train = rng.integers(0, 2, size=400)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 31, "verbose": -1},
        lgb.Dataset(X_train, y_train),
        num_boost_round=30,
    )
    fb = FastBooster(booster)

    X = rng.normal(size=(n_rows, 8)).astype(np.float64)
    if n_rows == 0:
        out = fb.predict(X)
        assert out.shape == (0,)
        assert out.dtype == np.float64
        return

    ref = booster.predict(X)
    np.testing.assert_allclose(fb.predict(X, kernel="quickscorer"), ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fb.predict(X, kernel="compact"), ref, rtol=1e-12, atol=1e-12)
