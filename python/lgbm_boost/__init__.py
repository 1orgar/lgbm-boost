from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._lgbm_boost_ext import (
    FastEnsembleScorer,
    detect_best_isa,
    is_isa_available,
    lower_threshold_to_f32,
)

__all__ = [
    "FastBooster",
    "FastEnsembleScorer",
    "detect_best_isa",
    "is_isa_available",
    "lower_threshold_to_f32",
]

_SUPPORTED_COLUMN_KINDS = frozenset({"f", "i", "u", "b"})


class FastBooster:
    """High-throughput SIMD batch scorer compatible with LightGBM Booster models.

    Supports loading directly from a ``lightgbm.Booster`` (or sklearn wrapper
    ``LGBMRegressor`` / ``LGBMClassifier`` / ``LGBMRanker``), a model file path,
    or a LightGBM model string.
    """

    def __init__(
        self,
        model: Any,
        *,
        qs32_block_trees: int = 64,
        qs64_block_trees: int = 64,
    ) -> None:
        if isinstance(model, FastEnsembleScorer):
            self._scorer = model
            return

        # Support sklearn-API estimators (LGBMClassifier, LGBMRegressor, LGBMRanker)
        if hasattr(model, "booster_"):
            model = model.booster_

        # Support lightgbm.Booster directly
        if hasattr(model, "model_to_string"):
            model_str = model.model_to_string()
            self._scorer = FastEnsembleScorer.from_model_string(
                model_str,
                qs32_block_trees=qs32_block_trees,
                qs64_block_trees=qs64_block_trees,
            )
            return

        if isinstance(model, Path):
            self._scorer = FastEnsembleScorer.from_model_file(
                str(model),
                qs32_block_trees=qs32_block_trees,
                qs64_block_trees=qs64_block_trees,
            )
            return

        if isinstance(model, str):
            # Distinguish inline model string vs file path
            if "\nTree=" in model or model.startswith("tree\n") or model.startswith("tree\r\n"):
                self._scorer = FastEnsembleScorer.from_model_string(
                    model,
                    qs32_block_trees=qs32_block_trees,
                    qs64_block_trees=qs64_block_trees,
                )
            else:
                self._scorer = FastEnsembleScorer.from_model_file(
                    model,
                    qs32_block_trees=qs32_block_trees,
                    qs64_block_trees=qs64_block_trees,
                )
            return

        raise TypeError(
            f"Unsupported model type {type(model)!r}. Pass a lightgbm.Booster, "
            "LGBMModel, file path, or model string."
        )

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        *,
        qs32_block_trees: int = 64,
        qs64_block_trees: int = 64,
    ) -> FastBooster:
        return cls(
            FastEnsembleScorer.from_model_file(
                str(filepath),
                qs32_block_trees=qs32_block_trees,
                qs64_block_trees=qs64_block_trees,
            )
        )

    @classmethod
    def from_string(
        cls,
        model_str: str,
        *,
        qs32_block_trees: int = 64,
        qs64_block_trees: int = 64,
    ) -> FastBooster:
        return cls(
            FastEnsembleScorer.from_model_string(
                model_str,
                qs32_block_trees=qs32_block_trees,
                qs64_block_trees=qs64_block_trees,
            )
        )

    @property
    def num_features(self) -> int:
        return self._scorer.num_features

    @property
    def num_outputs(self) -> int:
        return self._scorer.num_outputs

    @property
    def total_iterations(self) -> int:
        return self._scorer.total_iterations

    @property
    def total_trees(self) -> int:
        return self._scorer.total_trees

    @property
    def feature_names(self) -> list[str]:
        return self._scorer.feature_names

    def _allocate_or_validate_out(
        self, n_rows: int, out: np.ndarray | None
    ) -> np.ndarray:
        k = self._scorer.num_outputs
        expected_shape = (n_rows,) if k == 1 else (n_rows, k)
        if out is None:
            return np.empty(expected_shape, dtype=np.float32)
        if out.dtype != np.float32 or not out.flags.c_contiguous or out.shape != expected_shape:
            raise ValueError(
                f"out must be C-contiguous float32 ndarray with shape {expected_shape}, "
                f"got dtype={out.dtype}, shape={out.shape}"
            )
        return out

    def predict(
        self,
        data: Any,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = False,
        *,
        num_threads: int = 0,
        isa: str = "auto",
        kernel: str = "auto",
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score a batch of samples using SIMD tree evaluation.

        Accepts ``numpy.ndarray`` (row-major, Fortran-major, or strided views)
        and ``pandas.DataFrame`` (zero-copy per-column extraction without full
        matrix allocation).
        """
        # 1. Fast path for numpy.ndarray
        if isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ValueError(f"Expected 2D input array, got {arr.ndim}D")

            n_rows, n_cols = arr.shape
            out_arr = self._allocate_or_validate_out(n_rows, out)
            if n_rows == 0:
                return out_arr

            # Row-contiguous float32 (zero copy)
            if arr.dtype == np.float32 and (n_cols <= 1 or arr.strides[1] == 4):
                self._scorer.predict_f32(
                    arr,
                    out_arr,
                    start_iteration,
                    num_iteration,
                    raw_score,
                    num_threads,
                    isa,
                    kernel,
                )
                return out_arr

            # Row-contiguous float64 (zero copy, streamed by L2 tiles)
            if arr.dtype == np.float64 and (n_cols <= 1 or arr.strides[1] == 8):
                self._scorer.predict_f64(
                    arr,
                    out_arr,
                    start_iteration,
                    num_iteration,
                    raw_score,
                    num_threads,
                    isa,
                    kernel,
                )
                return out_arr

            # Fortran-order (column-contiguous) ndarray -> zero-copy columnar path!
            if (
                arr.strides[0] == arr.itemsize
                and arr.dtype.kind in _SUPPORTED_COLUMN_KINDS
                and arr.itemsize in (1, 2, 4, 8)
            ):
                cols = [arr[:, j] for j in range(n_cols)]
                self._scorer.predict_columnar(
                    cols,
                    out_arr,
                    start_iteration,
                    num_iteration,
                    raw_score,
                    num_threads,
                    isa,
                    kernel,
                )
                return out_arr

            # General fallback (e.g. non-contiguous or float16/object): cast to C-contiguous float32
            arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
            self._scorer.predict_f32(
                arr_f32,
                out_arr,
                start_iteration,
                num_iteration,
                raw_score,
                num_threads,
                isa,
                kernel,
            )
            return out_arr

        # 2. Zero-copy columnar path for pandas.DataFrame (avoids full float64 matrix copy!)
        if hasattr(data, "dtypes") and hasattr(data, "shape") and hasattr(data, "iloc"):
            n_rows, n_cols = data.shape
            out_arr = self._allocate_or_validate_out(n_rows, out)
            if n_rows == 0:
                return out_arr

            cols = []
            for j in range(n_cols):
                series = data.iloc[:, j]
                col_arr = series.to_numpy(copy=False)
                if (
                    not isinstance(col_arr, np.ndarray)
                    or not col_arr.flags.c_contiguous
                    or col_arr.dtype.kind not in _SUPPORTED_COLUMN_KINDS
                    or col_arr.itemsize not in (1, 2, 4, 8)
                ):
                    # Nullable Arrow/Extension integer/float column -> convert just this 1D column
                    col_arr = np.ascontiguousarray(series.to_numpy(dtype=np.float32, na_value=np.nan))
                cols.append(col_arr)

            self._scorer.predict_columnar(
                cols,
                out_arr,
                start_iteration,
                num_iteration,
                raw_score,
                num_threads,
                isa,
                kernel,
            )
            return out_arr

        # 3. Generic iterable / list-of-lists fallback
        arr_f32 = np.ascontiguousarray(data, dtype=np.float32)
        if arr_f32.ndim == 1:
            arr_f32 = arr_f32.reshape(1, -1)
        out_arr = self._allocate_or_validate_out(arr_f32.shape[0], out)
        if arr_f32.shape[0] > 0:
            self._scorer.predict_f32(
                arr_f32,
                out_arr,
                start_iteration,
                num_iteration,
                raw_score,
                num_threads,
                isa,
                kernel,
            )
        return out_arr
