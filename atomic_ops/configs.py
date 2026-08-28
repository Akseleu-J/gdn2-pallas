from __future__ import annotations
import dataclasses as dc
import os
import jax
import jax.lax as lax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


@dc.dataclass(frozen=True)
class KernelConfig:
    bt: int = 256
    bc: int = 128
    mb: int = 16
    clip: float = 1e4
    precision: lax.Precision = lax.Precision.HIGHEST

    @property
    def n_sub(self) -> int:
        return self.bt // self.bc

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb


KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4)
KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4)
KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3)
DEFAULT_CONFIG = KAGGLE_MEDIUM


def sanitize(x, config: KernelConfig = DEFAULT_CONFIG):
    c = config.clip
    return jnp.nan_to_num(jnp.clip(x, -c, c), nan=0.0, posinf=c, neginf=-c)


def sanitize_h0(h0, config: KernelConfig = DEFAULT_CONFIG):
    return sanitize(h0, config)


def clip_acc(x, config: KernelConfig = DEFAULT_CONFIG):
    return sanitize(x, config)


def _reshape_to_chunks(t, bsz, n_chunks, H, D, bt):
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t, bsz, n_chunks, bt, H, D):
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6


def _stage_diag(tag: str, x):
    if not _GDN2_FWD_DIAG:
        return x
    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    n_nonfinite = jnp.sum(jnp.logical_not(finite_mask))
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    def _report_nonfinite():
        jax.debug.print(
            "[GDN2-FWD-DIAG] non-finite at " + tag + ": n={n} max_abs_finite={m:.3e}",
            n=n_nonfinite, m=max_abs)

    def _report_large():
        jax.debug.print(
            "[GDN2-FWD-DIAG] suspiciously large at " + tag + ": max_abs={m:.3e}", m=max_abs)

    jax.lax.cond(
        jnp.logical_not(all_finite), _report_nonfinite,
        lambda: jax.lax.cond(max_abs > _LARGE_THRESHOLD, _report_large, lambda: None),
    )
    return x


def validate_inputs(q, k, v, w, b, g, scale, h0, config: KernelConfig):
    bsz, L, H, D = q.shape
    if D != 128:
        raise ValueError(f"Kernel assumes d_head=128 (MXU tile); got D={D}.")
    if L % config.bt != 0:
        raise ValueError(f"seq_len={L} must be divisible by bt={config.bt}.")
    for name, t in (("k", k), ("v", v), ("w", w), ("b", b), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(f"{name}.shape={t.shape} != q.shape={q.shape}")
    n_chunks = L // config.bt
    return bsz, L, H, D, n_chunks
