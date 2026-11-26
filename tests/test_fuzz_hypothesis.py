from __future__ import annotations

from hypothesis import given, settings, strategies as st
import lightgbm as lgb
import numpy as np

from lgbm_boost import FastBooster


@settings(max_examples=35, deadline=None)
@given(
    n_rows=st.integers(min_value=1, max_value=310),
    n_cols=st.integers(min_value=1, max_value=18),
    num_leaves=st.sampled_from([2, 7, 31, 32, 33, 63, 64, 75]),
    n_rounds=st.integers(min_value=1, max_value=25),
    objective=st.sampled_from(["regression", "binary", "multiclass"]),
    zero_as_missing=st.booleans(),
    use_missing=st.booleans(),
    nan_prob=st.floats(min_value=0.0, max_value=0.25),
    zero_prob=st.floats(min_value=0.0, max_value=0.25),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hypothesis_random_models_and_inputs(
    n_rows: int,
    n_cols: int,
    num_leaves: int,
    n_rounds: int,
    objective: str,
    zero_as_missing: bool,
    use_missing: bool,
    nan_prob: float,
    zero_prob: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    n_train = 160
    X_train = rng.normal(scale=10.0, size=(n_train, n_cols)).astype(np.float64)
    if use_missing and nan_prob > 0:
        X_train[rng.random(X_train.shape) < nan_prob] = np.nan
    if zero_prob > 0:
        X_train[rng.random(X_train.shape) < zero_prob] = 0.0

    params: dict = {
        "objective": objective,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "zero_as_missing": zero_as_missing,
        "use_missing": use_missing,
        "verbose": -1,
        "seed": int(seed % 10007),
    }
    if objective == "binary":
        y_train = rng.integers(0, 2, size=n_train)
    elif objective == "multiclass":
        params["num_class"] = 3
        y_train = rng.integers(0, 3, size=n_train)
    else:
        y_train = rng.normal(size=n_train).astype(np.float64)

    booster = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=n_rounds)
    fb = FastBooster(booster)

    X_test = rng.normal(scale=12.0, size=(n_rows, n_cols)).astype(np.float64)
    if nan_prob > 0:
        X_test[rng.random(X_test.shape) < nan_prob] = np.nan
    if zero_prob > 0:
        X_test[rng.random(X_test.shape) < zero_prob] = 0.0

    ref = booster.predict(X_test)
    pred_qs = fb.predict(X_test, kernel="quickscorer", num_threads=2)
    pred_ct = fb.predict(X_test, kernel="compact", num_threads=2)

    assert pred_qs.dtype == np.float64
    assert pred_ct.dtype == np.float64
    np.testing.assert_allclose(pred_qs, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred_ct, ref, rtol=1e-12, atol=1e-12)
