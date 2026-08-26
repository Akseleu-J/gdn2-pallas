from __future__ import annotations
import jax.numpy as jnp
from .utils import is_tpu_available
from .gdn2_fwd import gdn2_pallas_forward as _pallas_fwd
from .gdn2_pipeline import gdn2_pallas_forward_trainable as _pallas_trainable
from .reference import gdn2_chunked_wy_reference


def gdn2_forward(q, k, v, w, b, g, scale, h0=None, config=None):
    """
    Универсальный forward: Pallas на TPU, JAX reference на CPU/GPU.
    Drop-in replacement для gdn2_pallas_forward.
    """
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_fwd(q, k, v, w, b, g, scale, h0=h0)
    # Fallback: чистый JAX (медленнее, но работает везде)
    return gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size=256, h0=h0)


def gdn2_forward_trainable(q, k, v, w, b, g, scale, h0=None, config=None):
    """
    Универсальный trainable: Pallas fused backward на TPU,
    JAX reference (через jax.grad) на CPU/GPU.
    """
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_trainable(q, k, v, w, b, g, scale, h0=h0)
    # Fallback: чистый JAX. jax.grad сам разберётся с backward.
    return gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size=256, h0=h0)
