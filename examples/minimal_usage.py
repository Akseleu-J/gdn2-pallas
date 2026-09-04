"""
Minimal usage example for gdn2-pallas.

Install:
    pip install gdn2-pallas

Run:
    python minimal_usage.py
"""
import jax.numpy as jnp
from atomic_ops import gdn2_forward_trainable

# Shapes: (batch, seq_len, heads, d_head). d_head must be 128.
batch, seq_len, heads, d_head = 4, 2048, 6, 128
shape = (batch, seq_len, heads, d_head)

q = jnp.ones(shape, dtype=jnp.float32)
k = jnp.ones(shape, dtype=jnp.float32)
v = jnp.ones(shape, dtype=jnp.float32)
w = jnp.ones(shape, dtype=jnp.float32)
b = jnp.ones(shape, dtype=jnp.float32)
g = jnp.ones(shape, dtype=jnp.float32)  # log-decay

scale = d_head ** -0.5

# Forward + backward (custom_vjp). Auto-picks Pallas on TPU, JAX
# reference fallback elsewhere.
out, h_final = gdn2_forward_trainable(q, k, v, w, b, g, scale)

print("output:", out.shape)      # (4, 2048, 6, 128)
print("final state:", h_final.shape)  # (4, 6, 128, 128)
