from __future__ import annotations

import atexit
from contextlib import ExitStack
import ctypes
from importlib.resources import as_file, files
from pathlib import Path
import sys
from typing import Any

import numpy as np

__version__ = "0.2.0"

__all__ = [
    "FastBooster",
    "FastEnsembleScorer",
    "detect_best_isa",
    "is_isa_available",
]

# ============================================================================
# Load pure C ABI shared library via importlib.resources (editable & zip-safe)
# ============================================================================

_EXIT_STACK = ExitStack()
atexit.register(_EXIT_STACK.close)


def _load_shared_library() -> ctypes.CDLL:
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    pkg_files = files("lgbm_boost")
    candidates = [f"liblgbm_boost_c{suffix}"]
    if sys.platform == "win32":
        candidates.append("lgbm_boost_c.dll")

    last_err: Exception | None = None
    for name in candidates:
        try:
            traversable = pkg_files / name
            lib_path = _EXIT_STACK.enter_context(as_file(traversable))
            if lib_path.exists():
                return ctypes.CDLL(str(lib_path))
        except Exception as exc:  # pragma: no cover
            last_err = exc

    # Fallback relative to __file__ for direct source-tree execution
    pkg_dir = Path(__file__).resolve().parent
    for name in candidates:
        candidate_path = pkg_dir / name
        if candidate_path.exists():
            return ctypes.CDLL(str(candidate_path))

    raise RuntimeError(
        f"Failed to locate compiled lgbm_boost C shared library ({candidates})"
    ) from last_err


_LIB = _load_shared_library()

# Configure C ABI function signatures
_LIB.lgbm_boost_get_last_error.argtypes = []
_LIB.lgbm_boost_get_last_error.restype = ctypes.c_char_p

_LIB.lgbm_boost_detect_best_isa.argtypes = []
_LIB.lgbm_boost_detect_best_isa.restype = ctypes.c_char_p

_LIB.lgbm_boost_is_isa_available.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_LIB.lgbm_boost_is_isa_available.restype = ctypes.c_int

_LIB.lgbm_boost_create_from_string.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_LIB.lgbm_boost_create_from_string.restype = ctypes.c_int

_LIB.lgbm_boost_create_from_file.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_LIB.lgbm_boost_create_from_file.restype = ctypes.c_int

_LIB.lgbm_boost_free.argtypes = [ctypes.c_void_p]
_LIB.lgbm_boost_free.restype = None

_LIB.lgbm_boost_get_metadata.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
_LIB.lgbm_boost_get_metadata.restype = ctypes.c_int

_LIB.lgbm_boost_num_feature_names.argtypes = [ctypes.c_void_p]
_LIB.lgbm_boost_num_feature_names.restype = ctypes.c_uint64

_LIB.lgbm_boost_get_feature_name.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_LIB.lgbm_boost_get_feature_name.restype = ctypes.c_char_p

_LIB.lgbm_boost_predict_row_major_f32.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_void_p,  # data (const float*)
    ctypes.c_uint64,  # n_rows
    ctypes.c_uint64,  # n_cols
    ctypes.c_int64,   # row_stride_elems
    ctypes.c_void_p,  # out_f64 (double*)
    ctypes.c_void_p,  # out_f32 (float*)
    ctypes.c_int32,   # start_iteration
    ctypes.c_int32,   # num_iteration
    ctypes.c_int32,   # raw_score
    ctypes.c_int32,   # num_threads
    ctypes.c_int32,   # isa_id
    ctypes.c_int32,   # kernel_id
]
_LIB.lgbm_boost_predict_row_major_f32.restype = ctypes.c_int

_LIB.lgbm_boost_predict_row_major_f64.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_void_p,  # data (const double*)
    ctypes.c_uint64,  # n_rows
    ctypes.c_uint64,  # n_cols
    ctypes.c_int64,   # row_stride_elems
    ctypes.c_void_p,  # out_f64 (double*)
    ctypes.c_void_p,  # out_f32 (float*)
    ctypes.c_int32,   # start_iteration
    ctypes.c_int32,   # num_iteration
    ctypes.c_int32,   # raw_score
    ctypes.c_int32,   # num_threads
    ctypes.c_int32,   # isa_id
    ctypes.c_int32,   # kernel_id
]
_LIB.lgbm_boost_predict_row_major_f64.restype = ctypes.c_int

_LIB.lgbm_boost_predict_columnar.argtypes = [
    ctypes.c_void_p,                  # handle
    ctypes.POINTER(ctypes.c_void_p),  # col_ptrs
    ctypes.POINTER(ctypes.c_uint8),   # col_dtypes
    ctypes.c_uint64,                  # n_rows
    ctypes.c_uint64,                  # n_cols
    ctypes.c_void_p,                  # out_f64 (double*)
    ctypes.c_void_p,                  # out_f32 (float*)
    ctypes.c_int32,                   # start_iteration
    ctypes.c_int32,                   # num_iteration
    ctypes.c_int32,                   # raw_score
    ctypes.c_int32,                   # num_threads
    ctypes.c_int32,                   # isa_id
    ctypes.c_int32,                   # kernel_id
]
_LIB.lgbm_boost_predict_columnar.restype = ctypes.c_int

# ============================================================================
# Enum & dtype mappings
# ============================================================================

_ISA_IDS: dict[str, int] = {
    "": 0,
    "auto": 0,
    "scalar": 1,
    "avx2": 2,
    "avx512": 3,
}

_KERNEL_IDS: dict[str, int] = {
    "": 0,
    "auto": 0,
    "quickscorer": 1,
    "qs": 1,
    "compact": 2,
    "tree": 2,
}

# Must match fast_scorer::ColumnDType in include/fast_scorer/model.h
_COLUMN_DTYPE_CODES: dict[np.dtype[Any], int] = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
    np.dtype(np.int32): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.int16): 4,
    np.dtype(np.int8): 5,
    np.dtype(np.uint8): 6,
    np.dtype(np.bool_): 6,
    np.dtype(np.uint16): 7,
    np.dtype(np.uint32): 8,
}


def _check_status(code: int) -> None:
    if code == 0:
        return
    raw_msg = _LIB.lgbm_boost_get_last_error()
    msg = raw_msg.decode("utf-8", errors="replace") if raw_msg else "Unknown C++ error"
    if code == -1:
        raise ValueError(msg)
    raise RuntimeError(msg)


def _parse_isa_id(isa: str) -> int:
    try:
        return _ISA_IDS[isa]
    except KeyError:
        raise ValueError(
            f"Unknown SIMD ISA {isa!r}. Expected 'auto', 'scalar', 'avx2', or 'avx512'."
        ) from None


def _parse_kernel_id(kernel: str) -> int:
    try:
        return _KERNEL_IDS[kernel]
    except KeyError:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected 'auto', 'quickscorer', or 'compact'."
        ) from None


def detect_best_isa() -> str:
    raw = _LIB.lgbm_boost_detect_best_isa()
    return raw.decode("ascii") if raw else "scalar"


def is_isa_available(isa: str) -> bool:
    isa_id = _parse_isa_id(isa)
    avail = ctypes.c_int(0)
    _check_status(_LIB.lgbm_boost_is_isa_available(isa_id, ctypes.byref(avail)))
    return bool(avail.value)


# ============================================================================
# Low-level ctypes wrapper around C++ FastEnsembleScorer
# ============================================================================


class FastEnsembleScorer:
    """Low-level Python-ABI-independent wrapper around C++ FastEnsembleScorer."""

    _free_fn = _LIB.lgbm_boost_free

    def __init__(self, handle: ctypes.c_void_p) -> None:
        if not handle or not handle.value:
            raise ValueError("Invalid null FastEnsembleScorer handle")
        self._handle = handle

        nf = ctypes.c_int32()
        no = ctypes.c_int32()
        ti = ctypes.c_int32()
        tt = ctypes.c_int32()
        ml = ctypes.c_int32()
        q32 = ctypes.c_uint64()
        q64 = ctypes.c_uint64()
        dt = ctypes.c_uint64()
        _check_status(
            _LIB.lgbm_boost_get_metadata(
                self._handle,
                ctypes.byref(nf),
                ctypes.byref(no),
                ctypes.byref(ti),
                ctypes.byref(tt),
                ctypes.byref(ml),
                ctypes.byref(q32),
                ctypes.byref(q64),
                ctypes.byref(dt),
            )
        )
        self.num_features: int = int(nf.value)
        self.num_outputs: int = int(no.value)
        self.total_iterations: int = int(ti.value)
        self.total_trees: int = int(tt.value)
        self.max_num_leaves: int = int(ml.value)
        self.num_qs32_blocks: int = int(q32.value)
        self.num_qs64_blocks: int = int(q64.value)
        self.num_deep_trees: int = int(dt.value)

        n_names = _LIB.lgbm_boost_num_feature_names(self._handle)
        names: list[str] = []
        for i in range(n_names):
            raw_name = _LIB.lgbm_boost_get_feature_name(self._handle, i)
            names.append(raw_name.decode("utf-8", errors="replace") if raw_name else "")
        self.feature_names: list[str] = names

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        free_fn = getattr(self, "_free_fn", None)
        if handle and getattr(handle, "value", None) and free_fn is not None:
            free_fn(handle)
            self._handle = ctypes.c_void_p(0)

    @classmethod
    def from_model_string(
        cls,
        model_str: str,
        qs32_block_trees: int = 64,
        qs64_block_trees: int = 64,
    ) -> FastEnsembleScorer:
        encoded = model_str.encode("utf-8")
        handle = ctypes.c_void_p()
        _check_status(
            _LIB.lgbm_boost_create_from_string(
                encoded,
                len(encoded),
                int(qs32_block_trees),
                int(qs64_block_trees),
                ctypes.byref(handle),
            )
        )
        return cls(handle)

    @classmethod
    def from_model_file(
        cls,
        filepath: str,
        qs32_block_trees: int = 64,
        qs64_block_trees: int = 64,
    ) -> FastEnsembleScorer:
        encoded = os_fsencode(filepath)
        handle = ctypes.c_void_p()
        _check_status(
            _LIB.lgbm_boost_create_from_file(
                encoded,
                int(qs32_block_trees),
                int(qs64_block_trees),
                ctypes.byref(handle),
            )
        )
        return cls(handle)

    def _resolve_out_ptrs(self, n_rows: int, out: np.ndarray) -> tuple[int, int]:
        if not out.flags.c_contiguous:
            raise ValueError("out array must be C-contiguous")
        if out.size != n_rows * self.num_outputs:
            raise ValueError("Output buffer size mismatch")
        if out.dtype == np.float64:
            return out.ctypes.data, 0
        if out.dtype == np.float32:
            return 0, out.ctypes.data
        raise ValueError(f"Unsupported output dtype {out.dtype}; expected float64 or float32")

    def predict_f32(
        self,
        X: np.ndarray,
        out: np.ndarray,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = False,
        num_threads: int = 0,
        isa: str = "auto",
        kernel: str = "auto",
    ) -> None:
        if X.ndim != 2 or X.dtype != np.float32:
            raise ValueError("X must be a 2D float32 numpy array")
        n_rows, n_cols = X.shape
        if n_cols > 1 and X.strides[1] != 4:
            raise ValueError("X must have contiguous rows (strides[1] == 4)")
        row_stride_elems = (X.strides[0] // 4) if n_rows > 0 else n_cols
        out_f64_ptr, out_f32_ptr = self._resolve_out_ptrs(n_rows, out)

        _check_status(
            _LIB.lgbm_boost_predict_row_major_f32(
                self._handle,
                X.ctypes.data,
                n_rows,
                n_cols,
                row_stride_elems,
                out_f64_ptr,
                out_f32_ptr,
                int(start_iteration),
                int(num_iteration),
                1 if raw_score else 0,
                int(num_threads),
                _parse_isa_id(isa),
                _parse_kernel_id(kernel),
            )
        )

    def predict_f64(
        self,
        X: np.ndarray,
        out: np.ndarray,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = False,
        num_threads: int = 0,
        isa: str = "auto",
        kernel: str = "auto",
    ) -> None:
        if X.ndim != 2 or X.dtype != np.float64:
            raise ValueError("X must be a 2D float64 numpy array")
        n_rows, n_cols = X.shape
        if n_cols > 1 and X.strides[1] != 8:
            raise ValueError("X must have contiguous rows (strides[1] == 8)")
        row_stride_elems = (X.strides[0] // 8) if n_rows > 0 else n_cols
        out_f64_ptr, out_f32_ptr = self._resolve_out_ptrs(n_rows, out)

        _check_status(
            _LIB.lgbm_boost_predict_row_major_f64(
                self._handle,
                X.ctypes.data,
                n_rows,
                n_cols,
                row_stride_elems,
                out_f64_ptr,
                out_f32_ptr,
                int(start_iteration),
                int(num_iteration),
                1 if raw_score else 0,
                int(num_threads),
                _parse_isa_id(isa),
                _parse_kernel_id(kernel),
            )
        )

    def predict_columnar(
        self,
        columns: list[np.ndarray],
        out: np.ndarray,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = False,
        num_threads: int = 0,
        isa: str = "auto",
        kernel: str = "auto",
    ) -> None:
        n_cols = len(columns)
        if n_cols == 0:
            raise ValueError("Empty column list")

        col_ptrs = (ctypes.c_void_p * n_cols)()
        col_dtypes = (ctypes.c_uint8 * n_cols)()
        n_rows = 0

        for idx, arr in enumerate(columns):
            if arr.ndim != 1 or not arr.flags.c_contiguous:
                raise ValueError(f"Column {idx} must be a 1D C-contiguous numpy array")
            if idx == 0:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError("All columns must have identical row counts")
            dt_code = _COLUMN_DTYPE_CODES.get(arr.dtype)
            if dt_code is None:
                raise ValueError(f"Unsupported column dtype {arr.dtype} at column {idx}")
            col_ptrs[idx] = arr.ctypes.data
            col_dtypes[idx] = dt_code

        out_f64_ptr, out_f32_ptr = self._resolve_out_ptrs(n_rows, out)

        _check_status(
            _LIB.lgbm_boost_predict_columnar(
                self._handle,
                col_ptrs,
                col_dtypes,
                n_rows,
                n_cols,
                out_f64_ptr,
                out_f32_ptr,
                int(start_iteration),
                int(num_iteration),
                1 if raw_score else 0,
                int(num_threads),
                _parse_isa_id(isa),
                _parse_kernel_id(kernel),
            )
        )


def os_fsencode(path: str) -> bytes:
    if sys.platform == "win32":
        return path.encode("utf-8")
    import os

    return os.fsencode(path)


# ============================================================================
# High-level user-facing FastBooster API
# ============================================================================


class FastBooster:
    """High-throughput exact float64 SIMD batch scorer compatible with LightGBM Booster models.

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
        self,
        n_rows: int,
        out: np.ndarray | None,
        dtype: Any,
    ) -> np.ndarray:
        k = self._scorer.num_outputs
        expected_shape = (n_rows,) if k == 1 else (n_rows, k)
        if out is None:
            out_dtype = np.dtype(dtype)
            if out_dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
                raise ValueError(
                    f"dtype must be float64 or float32, got {out_dtype}"
                )
            return np.empty(expected_shape, dtype=out_dtype)

        if (
            out.dtype not in (np.dtype(np.float64), np.dtype(np.float32))
            or not out.flags.c_contiguous
            or out.shape != expected_shape
        ):
            raise ValueError(
                f"out must be C-contiguous float64 or float32 ndarray with shape {expected_shape}, "
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
        dtype: Any = np.float64,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score a batch of samples using exact float64 SIMD tree evaluation.

        Returns a ``float64`` NumPy array by default (matching ``lightgbm.Booster.predict``),
        or ``float32`` when ``dtype=np.float32`` is requested.
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
            out_arr = self._allocate_or_validate_out(n_rows, out, dtype)
            if n_rows == 0:
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

            # Fortran-order (column-contiguous) ndarray -> zero-copy columnar path!
            if arr.strides[0] == arr.itemsize and arr.dtype in _COLUMN_DTYPE_CODES:
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

            # General fallback (e.g. non-contiguous or float16/uint64/object): cast to C-contiguous float64
            arr_f64 = np.ascontiguousarray(arr, dtype=np.float64)
            self._scorer.predict_f64(
                arr_f64,
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
            out_arr = self._allocate_or_validate_out(n_rows, out, dtype)
            if n_rows == 0:
                return out_arr

            cols = []
            for j in range(n_cols):
                series = data.iloc[:, j]
                col_arr = series.to_numpy(copy=False)
                if (
                    not isinstance(col_arr, np.ndarray)
                    or not col_arr.flags.c_contiguous
                    or col_arr.dtype not in _COLUMN_DTYPE_CODES
                ):
                    # Nullable Arrow/Extension integer/float column -> convert just this 1D column to float64
                    col_arr = np.ascontiguousarray(series.to_numpy(dtype=np.float64, na_value=np.nan))
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
        arr_f64 = np.ascontiguousarray(data, dtype=np.float64)
        if arr_f64.ndim == 1:
            arr_f64 = arr_f64.reshape(1, -1)
        out_arr = self._allocate_or_validate_out(arr_f64.shape[0], out, dtype)
        if arr_f64.shape[0] > 0:
            self._scorer.predict_f64(
                arr_f64,
                out_arr,
                start_iteration,
                num_iteration,
                raw_score,
                num_threads,
                isa,
                kernel,
            )
        return out_arr
