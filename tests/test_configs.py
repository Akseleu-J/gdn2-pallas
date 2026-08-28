import jax.numpy as jnp
from atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, DEFAULT_CONFIG, sanitize

def test_default_config_is_medium():
    assert DEFAULT_CONFIG == KAGGLE_MEDIUM

def test_sanitize_clips_and_removes_nan():
    x = jnp.array([float("nan"), float("inf"), -float("inf"), 5.0])
    out = sanitize(x, KAGGLE_SMALL)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert float(out[3]) == 5.0

def test_n_sub_n_micro_consistency():
    for cfg in (KAGGLE_SMALL, KAGGLE_MEDIUM):
        assert cfg.bt % cfg.bc == 0
        assert cfg.bc % cfg.mb == 0
