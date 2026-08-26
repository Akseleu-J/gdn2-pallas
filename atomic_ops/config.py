"""
Global configuration, validation, and sanitization utilities.
Single source of truth for all constants used across kernels.
"""
from __future__ import annotations

import dataclasses as dc
import os
from typing import Tuple

import jax
import jax.numpy as jnp


@dc.dataclass(frozen=True)
class KernelConfig:
    """Production configuration for GDN-2 Pallas kernels."""
    bt: int = 256          # chunk size
    bc: int = 128          # sub-block size
    mb: int = 16           # micro-block for WY solve
    clip: float = 1e4      # global clip magnitude
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST

    @property
    def n_sub(self) -> int:
        return self.bt // self.bc

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb

    def __post_init__(self):
        assert self.bt % self.bc == 0, f"bt={self.bt} must be divisible by bc={self.bc}"
        assert self.bc % self.mb == 0, f"bc={self.bc} must be divisible by mb={self.mb}"


# Default singleton — can be overridden per-call
DEFAULT_CONFIG = KernelConfig()

# Diagnostic env var
_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6


def sanitize(x: jnp.ndarray, clip: float | None = None) -> jnp.ndarray:
    """Clip + nan_to_num. clip=None uses config default."""
    c = DEFAULT_CONFIG.clip if clip is None else clip
    return jnp.nan_to_num(jnp.clip(x, -c, c), nan=0.0, posinf=c, neginf=-c)


def sanitize_h0(h0: jnp.ndarray, clip: float | None = None) -> jnp.ndarray:
    """Sanitize recurrent state h0."""
    return sanitize(h0, clip)


def clip_acc(x: jnp.ndarray, clip: float | None = None) -> jnp.ndarray:
    """Per-iteration accumulator clip (B4-style)."""
    return sanitize(x, clip)


def _stage_diag(tag: str, x: jnp.ndarray) -> jnp.ndarray:
    """Diagnostic no-op unless GDN2_FWD_DIAG=1."""
    if not _GDN2_FWD_DIAG:
        return x
    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    n_nonfinite = jnp.sum(jnp.logical_not(finite_mask))
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    def _report_nonfinite():
        jax.debug.print(
            "[GDN2-FWD-DIAG] WARNING non-finite at {tag}: n_nonfinite={n} max_abs(finite)={m:.3e}",
            tag=tag, n=n_nonfinite, m=max_abs,
        )

    def _report_large():
        jax.debug.print(
            "[GDN2-FWD-DIAG] SUSPICIOUS large-but-finite at {tag}: max_abs={m:.3e}",
            tag=tag, m=max_abs,
        )

    jax.lax.cond(
        jnp.logical_not(all_finite),
        _report_nonfinite,
        lambda: jax.lax.cond(max_abs > _LARGE_THRESHOLD, _report_large, lambda: None),
    )
    return x


def validate_inputs(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    g: jnp.ndarray,
    scale: float,
    h0: jnp.ndarray | None,
    config: KernelConfig,
) -> Tuple[int, int, int, int, int]:
    """Validate all inputs and return (bsz, L, H, D, n_chunks)."""
    bsz, L, H, D = q.shape
    for name, t in (("k", k), ("v", v), ("w", w), ("b", b), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(f"{name} shape {t.shape} != q shape {q.shape}")
    if D != 128:
        raise ValueError(f"Only d_head=128 supported (MXU tile); got D={D}")
    if L % config.bt != 0:
        raise ValueError(f"seq_len={L} must be divisible by bt={config.bt}")
    if h0 is not None and h0.shape != (bsz, H, D, D):
        raise ValueError(f"h0 shape {h0.shape} != ({bsz}, {H}, {D}, {D})")
    n_chunks = L // config.bt
    return bsz, L, H, D, n_chunks
