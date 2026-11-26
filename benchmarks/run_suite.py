#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from lgbm_boost import FastBooster, detect_best_isa


def median_ms(fn, repeats: int = 5) -> tuple[float, np.ndarray]:
    out = fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lgbm-boost (exact float64) vs stock LightGBM")
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000, 10_000, 100_000, 500_000])
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--leaves", type=int, nargs="+", default=[31, 63, 127])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--objective", type=str, default="binary", choices=["regression", "binary", "multiclass"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"Platform          : {platform.platform()} ({platform.machine()})")
    print(f"Detected SIMD ISA : {detect_best_isa()}")
    print(f"LightGBM version  : {lgb.__version__}")
    print("-" * 110)

    rng = np.random.default_rng(42)
    n_train = 15_000
    X_train = rng.normal(size=(n_train, args.features)).astype(np.float64)
    X_train[rng.random(X_train.shape) < 0.05] = np.nan
    if args.objective == "binary":
        y_train = rng.integers(0, 2, size=n_train)
    elif args.objective == "multiclass":
        y_train = rng.integers(0, 4, size=n_train)
    else:
        y_train = rng.normal(size=n_train).astype(np.float64)

    for num_leaves in args.leaves:
        params: dict = {
            "objective": args.objective,
            "num_leaves": num_leaves,
            "verbose": -1,
            "num_threads": max(args.threads),
        }
        if args.objective == "multiclass":
            params["num_class"] = 4

        booster = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=args.trees)
        fb = FastBooster(booster)

        print(f"\n=== Model: objective={args.objective}, trees={fb.total_trees}, num_leaves={num_leaves}, features={args.features} (output=float64) ===")
        print(f"{'Rows':>9} | {'Input':>10} | {'Thr':>3} | {'LightGBM (ms)':>13} | {'FastQS (ms)':>11} | {'Speedup':>8} | {'Compact (ms)':>12} | {'MaxDiff':>9}")
        print("-" * 110)

        for n_rows in args.rows:
            X64 = rng.normal(size=(n_rows, args.features)).astype(np.float64)
            X64[rng.random(X64.shape) < 0.05] = np.nan
            X32 = X64.astype(np.float32)

            for thr in args.threads:
                lgb_ms, ref64 = median_ms(lambda: booster.predict(X64, num_threads=thr), repeats=args.repeats)
                qs_ms, p_qs = median_ms(lambda: fb.predict(X64, num_threads=thr, kernel="quickscorer"), repeats=args.repeats)
                ct_ms, p_ct = median_ms(lambda: fb.predict(X64, num_threads=thr, kernel="compact"), repeats=args.repeats)
                max_diff = float(np.max(np.abs(p_qs - ref64)))
                best_ms = min(qs_ms, ct_ms)
                print(
                    f"{n_rows:9d} | {'ndarray64':>10} | {thr:3d} | {lgb_ms:13.2f} | {qs_ms:11.2f} | {lgb_ms / best_ms:7.2f}x | {ct_ms:12.2f} | {max_diff:9.2e}"
                )

            # Also measure float32 input -> float64 output on max threads
            thr = max(args.threads)
            lgb32_ms, ref32 = median_ms(lambda: booster.predict(X32, num_threads=thr), repeats=args.repeats)
            qs32_ms, p_qs32 = median_ms(lambda: fb.predict(X32, num_threads=thr, kernel="quickscorer"), repeats=args.repeats)
            ct32_ms, _ = median_ms(lambda: fb.predict(X32, num_threads=thr, kernel="compact"), repeats=args.repeats)
            diff32 = float(np.max(np.abs(p_qs32 - ref32)))
            best32_ms = min(qs32_ms, ct32_ms)
            print(
                f"{n_rows:9d} | {'ndarray32':>10} | {thr:3d} | {lgb32_ms:13.2f} | {qs32_ms:11.2f} | {lgb32_ms / best32_ms:7.2f}x | {ct32_ms:12.2f} | {diff32:9.2e}"
            )

            # Also benchmark mixed-dtype pandas.DataFrame on max thread count
            df = pd.DataFrame(X64, columns=[f"f_{i}" for i in range(args.features)])
            for i in range(1, args.features, 2):
                df[f"f_{i}"] = df[f"f_{i}"].astype(np.float32)
            lgb_df_ms, ref_df = median_ms(lambda: booster.predict(df, num_threads=thr), repeats=args.repeats)
            qs_df_ms, p_df = median_ms(lambda: fb.predict(df, num_threads=thr), repeats=args.repeats)
            diff_df = float(np.max(np.abs(p_df - ref_df)))
            print(
                f"{n_rows:9d} | {'DataFrame':>10} | {thr:3d} | {lgb_df_ms:13.2f} | {qs_df_ms:11.2f} | {lgb_df_ms / qs_df_ms:7.2f}x | {'-':>12} | {diff_df:9.2e}"
            )


if __name__ == "__main__":
    main()
