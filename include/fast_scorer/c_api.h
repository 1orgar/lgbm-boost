#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LGBM_BOOST_API __declspec(dllexport)
#else
#define LGBM_BOOST_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* LGBMBoostHandle;

// Status codes:
//   0 : OK
//  -1 : Invalid argument (maps to Python ValueError)
//  -2 : Runtime error (maps to Python RuntimeError)

LGBM_BOOST_API const char* lgbm_boost_get_last_error(void);
LGBM_BOOST_API const char* lgbm_boost_detect_best_isa(void);
LGBM_BOOST_API int lgbm_boost_is_isa_available(int isa_id, int* out_available);

LGBM_BOOST_API int lgbm_boost_create_from_string(
    const char* model_str,
    uint64_t model_len,
    uint32_t qs32_block_trees,
    uint32_t qs64_block_trees,
    LGBMBoostHandle* out_handle
);

LGBM_BOOST_API int lgbm_boost_create_from_file(
    const char* filepath,
    uint32_t qs32_block_trees,
    uint32_t qs64_block_trees,
    LGBMBoostHandle* out_handle
);

LGBM_BOOST_API void lgbm_boost_free(LGBMBoostHandle handle);

LGBM_BOOST_API int lgbm_boost_get_metadata(
    LGBMBoostHandle handle,
    int32_t* num_features,
    int32_t* num_outputs,
    int32_t* total_iterations,
    int32_t* total_trees,
    int32_t* max_num_leaves,
    uint64_t* num_qs32_blocks,
    uint64_t* num_qs64_blocks,
    uint64_t* num_deep_trees
);

LGBM_BOOST_API uint64_t lgbm_boost_num_feature_names(LGBMBoostHandle handle);
LGBM_BOOST_API const char* lgbm_boost_get_feature_name(LGBMBoostHandle handle, uint64_t idx);

LGBM_BOOST_API int lgbm_boost_predict_row_major_f32(
    LGBMBoostHandle handle,
    const float* data,
    uint64_t n_rows,
    uint64_t n_cols,
    int64_t row_stride_elems,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
);

LGBM_BOOST_API int lgbm_boost_predict_row_major_f64(
    LGBMBoostHandle handle,
    const double* data,
    uint64_t n_rows,
    uint64_t n_cols,
    int64_t row_stride_elems,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
);

LGBM_BOOST_API int lgbm_boost_predict_columnar(
    LGBMBoostHandle handle,
    const void* const* col_ptrs,
    const uint8_t* col_dtypes,
    uint64_t n_rows,
    uint64_t n_cols,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
);

#ifdef __cplusplus
} // extern "C"
#endif
