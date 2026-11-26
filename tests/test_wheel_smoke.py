from __future__ import annotations

import numpy as np

from lgbm_boost import FastBooster, detect_best_isa, is_isa_available

SAMPLE_BINARY_MODEL = """tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=3
objective=binary sigmoid:1
feature_names=f0 f1 f2 f3
feature_infos=[-3.0:3.0] [-3.0:3.0] [-3.0:3.0] [-3.0:3.0]
tree_sizes=400 400

Tree=0
num_leaves=4
num_cat=0
split_feature=0 1 2
split_gain=5.0 3.0 2.0
threshold=0.0 -0.5 0.5
decision_type=10 8 2
left_child=1 -2 -3
right_child=2 -1 -4
leaf_value=-0.5 0.25 -0.1 0.75
leaf_weight=10 10 10 10
leaf_count=10 10 10 10
internal_value=0 0 0
internal_weight=40 20 20
internal_count=40 20 20
is_linear=0
shrinkage=1

Tree=1
num_leaves=3
num_cat=0
split_feature=3 0
split_gain=4.0 2.0
threshold=1.0 -1.0
decision_type=6 10
left_child=1 -2
right_child=-1 -3
leaf_value=0.4 -0.3 0.2
leaf_weight=15 15 10
leaf_count=15 15 10
internal_value=0 0
internal_weight=40 25
internal_count=40 25
is_linear=0
shrinkage=1

end of trees
"""


def test_wheel_standalone_binary_and_isa_consistency() -> None:
    fb = FastBooster.from_string(SAMPLE_BINARY_MODEL)
    assert fb.num_features == 4
    assert fb.total_trees == 2
    assert fb.feature_names == ["f0", "f1", "f2", "f3"]

    rng = np.random.default_rng(42)
    X = rng.normal(size=(257, 4)).astype(np.float32)
    X[0, 0] = np.nan
    X[1, 3] = 0.0

    p_auto = fb.predict(X, num_threads=2)
    p_scalar_qs = fb.predict(X, isa="scalar", kernel="quickscorer", num_threads=1)
    p_scalar_ct = fb.predict(X, isa="scalar", kernel="compact", num_threads=1)

    assert p_auto.shape == (257,)
    assert np.all((p_auto > 0.0) & (p_auto < 1.0))
    np.testing.assert_allclose(p_auto, p_scalar_qs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p_auto, p_scalar_ct, rtol=1e-6, atol=1e-6)

    best = detect_best_isa()
    assert best in ("scalar", "avx2", "avx512")
    if is_isa_available("avx2"):
        p_avx2 = fb.predict(X, isa="avx2", kernel="quickscorer")
        np.testing.assert_allclose(p_avx2, p_scalar_qs, rtol=1e-6, atol=1e-6)
    if is_isa_available("avx512"):
        p_avx512 = fb.predict(X, isa="avx512", kernel="quickscorer")
        np.testing.assert_allclose(p_avx512, p_scalar_qs, rtol=1e-6, atol=1e-6)
