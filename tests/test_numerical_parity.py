from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lgbm_boost import FastBooster, detect_best_isa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
X86_CLI = PROJECT_ROOT / "build_x86" / "fast_scorer_cli"


@pytest.mark.parametrize(
    ("objective", "extra_params"),
    [
        ("regression", {}),
        ("regression_l1", {}),
        ("binary", {}),
        ("binary", {"sigmoid": 1.7}),
        ("cross_entropy", {}),
        ("multiclass", {"num_class": 4}),
        ("multiclassova", {"num_class": 3}),
        ("poisson", {}),
        ("regression", {"boosting": "rf", "bagging_freq": 1, "bagging_fraction": 0.8}),
    ],
)
@pytest.mark.parametrize("num_leaves", [2, 15, 31, 32, 33, 63, 64, 65, 127])
def test_objectives_and_leaf_tiers(objective: str, extra_params: dict, num_leaves: int) -> None:
    rng = np.random.default_rng(42 + num_leaves)
    n_rows, n_cols = 600, 20
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    X[rng.random(X.shape) < 0.06] = np.nan

    if objective in ("binary", "cross_entropy"):
        y = rng.integers(0, 2, size=n_rows)
        if objective == "cross_entropy":
            y = rng.uniform(0.0, 1.0, size=n_rows)
    elif objective in ("multiclass", "multiclassova"):
        y = rng.integers(0, extra_params["num_class"], size=n_rows)
    elif objective == "poisson":
        y = rng.poisson(lam=2.5, size=n_rows).astype(np.float64)
    else:
        y = rng.normal(size=n_rows).astype(np.float64)

    params = {
        "objective": objective,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 2,
        "verbose": -1,
        "seed": 123,
        **extra_params,
    }
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=40)
    fb = FastBooster(booster)

    for raw_score in (False, True):
        ref = booster.predict(X, raw_score=raw_score)
        for kernel in ("auto", "quickscorer", "compact"):
            pred = fb.predict(X, raw_score=raw_score, kernel=kernel, num_threads=2)
            assert pred.dtype == np.float64
            assert pred.shape == ref.shape
            np.testing.assert_allclose(pred, ref, rtol=1e-12, atol=1e-12)


def test_constant_single_leaf_and_mixed_depth_ensemble() -> None:
    rng = np.random.default_rng(999)
    X = rng.normal(size=(300, 12)).astype(np.float64)
    y = rng.normal(size=300).astype(np.float64)

    # First train normal 31-leaf trees, then continue with huge min_data_in_leaf -> 1-leaf constant trees,
    # then continue with 90-leaf deep trees to exercise hybrid QS32 + CompactTree + ConstantTree simultaneously!
    b1 = lgb.train({"objective": "regression", "num_leaves": 31, "min_data_in_leaf": 5, "verbose": -1}, lgb.Dataset(X, y), num_boost_round=20)
    b2 = lgb.train({"objective": "regression", "num_leaves": 31, "min_data_in_leaf": 1000, "verbose": -1}, lgb.Dataset(X, y), num_boost_round=5, init_model=b1)
    b3 = lgb.train({"objective": "regression", "num_leaves": 90, "min_data_in_leaf": 2, "verbose": -1}, lgb.Dataset(X, y), num_boost_round=15, init_model=b2)

    fb = FastBooster(b3)
    ref_full = b3.predict(X)
    pred_full = fb.predict(X)
    assert pred_full.dtype == np.float64
    np.testing.assert_allclose(pred_full, ref_full, rtol=1e-12, atol=1e-12)

    # Test partial iteration slices across the 3 regimes
    for start_it, num_it in [(0, 10), (15, 10), (20, 5), (18, 15), (25, -1)]:
        ref_slice = b3.predict(X, start_iteration=start_it, num_iteration=num_it)
        pred_slice = fb.predict(X, start_iteration=start_it, num_iteration=num_it)
        np.testing.assert_allclose(pred_slice, ref_slice, rtol=1e-12, atol=1e-12)


def test_input_layouts_and_output_dtypes_parity() -> None:
    rng = np.random.default_rng(2026)
    X_f64 = rng.normal(size=(555, 16)).astype(np.float64)
    X_f64[rng.random(X_f64.shape) < 0.05] = np.nan
    y = rng.integers(0, 3, size=555)

    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "num_leaves": 31, "verbose": -1},
        lgb.Dataset(X_f64, y),
        num_boost_round=50,
    )
    fb = FastBooster(booster)
    ref64 = booster.predict(X_f64)

    # 1. C-contiguous float64 (default float64 output)
    p_c64 = fb.predict(X_f64)
    assert p_c64.dtype == np.float64
    np.testing.assert_allclose(p_c64, ref64, rtol=1e-12, atol=1e-12)

    # 2. C-contiguous float32 input -> exact parity with booster.predict(X_f32)
    X_f32 = X_f64.astype(np.float32)
    ref32_in = booster.predict(X_f32)
    p_c32_in = fb.predict(X_f32)
    assert p_c32_in.dtype == np.float64
    np.testing.assert_allclose(p_c32_in, ref32_in, rtol=1e-12, atol=1e-12)

    # 3. Optional dtype=np.float32 output
    p_out32 = fb.predict(X_f64, dtype=np.float32)
    assert p_out32.dtype == np.float32
    np.testing.assert_allclose(p_out32, ref64.astype(np.float32), rtol=1e-6, atol=1e-6)

    # 4. Fortran-contiguous float64 & float32
    X_f_order64 = np.asfortranarray(X_f64)
    np.testing.assert_allclose(fb.predict(X_f_order64), ref64, rtol=1e-12, atol=1e-12)
    X_f_order32 = np.asfortranarray(X_f32)
    np.testing.assert_allclose(fb.predict(X_f_order32), ref32_in, rtol=1e-12, atol=1e-12)

    # 5. Strided row-slice X[::3]
    np.testing.assert_allclose(fb.predict(X_f64[::3]), ref64[::3], rtol=1e-12, atol=1e-12)

    # 6. Mixed-dtype pandas.DataFrame (float64, float32, int32, int64, uint8)
    df = pd.DataFrame(X_f64, columns=[f"f{i}" for i in range(16)])
    df["f0"] = df["f0"].astype(np.float32)
    df["f1"] = np.arange(555, dtype=np.int32)
    df["f2"] = np.arange(555, dtype=np.int64)
    df["f3"] = (np.arange(555) % 7).astype(np.uint8)
    ref_df = booster.predict(df)
    pred_df = fb.predict(df)
    assert pred_df.dtype == np.float64
    np.testing.assert_allclose(pred_df, ref_df, rtol=1e-12, atol=1e-12)

    # 7. Single 1D row & preallocated out= buffers (both float64 and float32)
    p_1d = fb.predict(X_f64[0])
    assert p_1d.dtype == np.float64
    np.testing.assert_allclose(p_1d, ref64[:1], rtol=1e-12, atol=1e-12)

    out_buf64 = np.empty((555, 3), dtype=np.float64)
    ret64 = fb.predict(X_f64, out=out_buf64)
    assert ret64 is out_buf64
    np.testing.assert_allclose(out_buf64, ref64, rtol=1e-12, atol=1e-12)

    out_buf32 = np.empty((555, 3), dtype=np.float32)
    ret32 = fb.predict(X_f64, out=out_buf32)
    assert ret32 is out_buf32
    np.testing.assert_allclose(out_buf32, ref64.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_sklearn_wrapper_and_file_loading() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 10)).astype(np.float64)
    y = rng.integers(0, 2, size=300)

    clf = lgb.LGBMClassifier(n_estimators=35, num_leaves=25, verbose=-1, random_state=7)
    clf.fit(X, y)

    fb_est = FastBooster(clf)
    ref_proba = clf.predict_proba(X)[:, 1]
    pred_est = fb_est.predict(X)
    assert pred_est.dtype == np.float64
    np.testing.assert_allclose(pred_est, ref_proba, rtol=1e-12, atol=1e-12)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.txt"
        clf.booster_.save_model(str(model_path))
        fb_file = FastBooster.from_file(model_path)
        np.testing.assert_allclose(fb_file.predict(X), ref_proba, rtol=1e-12, atol=1e-12)
    assert isinstance(detect_best_isa(), str)


@pytest.mark.skipif(not X86_CLI.exists(), reason="x86_64 CLI binary not built")
@pytest.mark.parametrize(
    ("objective", "num_class", "num_leaves", "zero_as_missing"),
    [
        ("binary", 1, 31, False),
        ("binary", 1, 32, True),
        ("multiclass", 3, 55, True),
        ("regression", 1, 64, False),
        ("regression", 1, 110, True),
    ],
)
def test_x86_64_avx2_intrinsics_parity(
    objective: str, num_class: int, num_leaves: int, zero_as_missing: bool
) -> None:
    rng = np.random.default_rng(314 + num_leaves)
    n_rows, n_cols = 1037, 24
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    X[rng.random(X.shape) < 0.08] = np.nan
    X[rng.random(X.shape) < 0.08] = 0.0

    if objective == "binary":
        y = rng.integers(0, 2, size=n_rows)
    elif objective == "multiclass":
        y = rng.integers(0, num_class, size=n_rows)
    else:
        y = rng.normal(size=n_rows).astype(np.float64)

    params = {
        "objective": objective,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 2,
        "zero_as_missing": zero_as_missing,
        "verbose": -1,
    }
    if num_class > 1:
        params["num_class"] = num_class

    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=80)
    ref = np.ascontiguousarray(booster.predict(X), dtype=np.float64)

    env = os.environ.copy()
    env["ROSETTA_ADVERTISE_AVX"] = "1"

    with tempfile.TemporaryDirectory() as d:
        m_path = f"{d}/m.txt"
        x_path = f"{d}/x.bin"
        r_path = f"{d}/ref.bin"
        booster.save_model(m_path)
        X.tofile(x_path)
        ref.tofile(r_path)

        for kernel in ("quickscorer", "compact"):
            proc = subprocess.run(
                [
                    str(X86_CLI),
                    m_path,
                    x_path,
                    r_path,
                    str(n_rows),
                    str(n_cols),
                    str(num_class),
                    "0",
                    "avx2",
                    kernel,
                    "4",
                    "1e-12",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"AVX2 CLI failed: {proc.stderr}\n{proc.stdout}"
            assert "requested_isa=avx2" in proc.stdout
