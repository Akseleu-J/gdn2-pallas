"""
benchmark_gdn2_vs_jax_reference.py

THREE-TIER comparison for GDN-2:

  (OLD)  associative_scan-based recurrence -- gdn2_recurrence_safe, the
         implementation that was actually running in model.py BEFORE the
         chunked-WY/Pallas pipeline existed (copied verbatim from the
         project's own benchmark_gdn2.py). Chunk-parallel via
         jax.lax.associative_scan over (M,C) linear-recurrence pairs,
         custom_vjp with grad clipping. This is the honest "what we had
         before" baseline, not a strawman.

  (JAX_REF) gdn2_wy_reference.gdn2_chunked_wy_reference -- pure-JAX
         chunked WY algorithm (same chunk-parallel structure as the Pallas
         path: block decay + matrix inverse via lax.scan), no Pallas
         kernels. The project's own validated ground-truth for backward --
         an honest "plain JAX ceiling" for THIS algorithm.

  (PALLAS) kernel_trainable_B6.gdn2_pallas_forward_trainable -- honest
         fused forward (Kernel A/B/C/D) + fused backward (B1-B5), the path
         actually used in model.py today.

For EACH of the three we time:
  1. forward only
  2. backward only (pure vjp-application time; forward computed once via
     jax.vjp outside the timed call)
  3. forward+backward together (jax.value_and_grad, single jit'd call --
     what a real train step actually pays)

This gives two speedup numbers per config: OLD->JAX_REF (algorithmic win:
associative_scan vs chunked WY, both plain JAX) and JAX_REF->PALLAS
(kernel-fusion win: chunked WY vs the same algorithm hand-fused into
Pallas), plus the compounded OLD->PALLAS number.

Run on TPU:
    python benchmark_gdn2_vs_jax_reference.py
"""
import time
from functools import partial

import jax
import jax.numpy as jnp

from atomic_ops.kernel_trainable_B6 import gdn2_pallas_forward_trainable
from atomic_ops.gdn2_wy_reference import gdn2_chunked_wy_reference
from atomic_ops.kernel_a_scores import BT as PALLAS_BT


# ============================================================
# (OLD) verbatim associative_scan-based GDN-2 core, copied from the
# project's own benchmark_gdn2.py -- this is what model.py actually ran
# before the chunked-WY/Pallas pipeline replaced it.
# ============================================================
def _gdn2_recurrence_impl(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    def _combine(state1, state2):
        m1, c1 = state1
        m2, c2 = state2
        m_new = m2 @ m1
        fro_norm = jnp.sqrt(jnp.sum(jnp.square(m_new), axis=(-2, -1), keepdims=True))
        scale = jnp.minimum(1.0, 1.0 / (fro_norm + 1e-6))
        m_new = m_new * scale
        c_new = m2 @ c1 + c2
        c_new = jnp.nan_to_num(c_new, nan=0.0, posinf=1e4, neginf=-1e4)
        return m_new, c_new

    def _to_chunks(t):
        t = t.reshape(b, num_chunks, chunk_size, n_heads, d_head)
        return jnp.moveaxis(t, 1, 0)

    k_ch, ea_ch, z_ch, alpha_ch, q_ch = map(_to_chunks, (k, ea, z, alpha, q))

    eye_bh = jnp.broadcast_to(jnp.eye(d_head, dtype=dtype), (b, n_heads, d_head, d_head))
    zero_bh = jnp.zeros((b, n_heads, d_head, d_head), dtype=dtype)

    def _chunk_step(carry, chunk_inputs):
        carry_M, carry_S = carry
        k_c, ea_c, z_c, alpha_c, q_c = chunk_inputs
        eye = jnp.eye(d_head, dtype=dtype)[None, None, None, :, :]
        M_c = eye * alpha_c[:, :, :, None, :] - k_c[:, :, :, :, None] @ ea_c[:, :, :, None, :]
        C_c = k_c[:, :, :, :, None] @ z_c[:, :, :, None, :]
        P_local, S_local = jax.lax.associative_scan(_combine, (M_c, C_c), axis=1)
        global_M = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_M)
        global_S = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_S) + S_local
        global_S = jnp.nan_to_num(global_S, nan=0.0, posinf=1e4, neginf=-1e4)
        out_c = jnp.einsum("bchij,bchi->bchj", global_S, q_c)
        new_carry = (global_M[:, -1], global_S[:, -1])
        return new_carry, out_c

    _chunk_step = jax.checkpoint(_chunk_step)
    _, out_chunks = jax.lax.scan(
        _chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch)
    )
    return jnp.moveaxis(out_chunks, 0, 1).reshape(b, num_chunks * chunk_size, n_heads * d_head)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def gdn2_recurrence_safe(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    return _gdn2_recurrence_impl(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype)


def _gdn2_recurrence_safe_fwd(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    primal_fn = lambda k, ea, z, alpha, q: _gdn2_recurrence_impl(
        k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype
    )
    out, vjp_fn = jax.vjp(primal_fn, k, ea, z, alpha, q)
    return out, vjp_fn


def _gdn2_recurrence_safe_bwd(chunk_size, num_chunks, b, n_heads, d_head, dtype, vjp_fn, g):
    grads = vjp_fn(g)
    _GRAD_CLIP = 1e3
    safe_grads = tuple(
        jnp.nan_to_num(jnp.clip(gr, -_GRAD_CLIP, _GRAD_CLIP), nan=0.0, posinf=_GRAD_CLIP, neginf=-_GRAD_CLIP)
        for gr in grads
    )
    return safe_grads


gdn2_recurrence_safe.defvjp(_gdn2_recurrence_safe_fwd, _gdn2_recurrence_safe_bwd)


def old_associative_scan_forward(q, k, v, w, b, g, scale, h0, chunk_size, n_heads, d_head, bsz):
    # h0 ignored -- the old associative_scan path never took an initial
    # cross-call state (always starts from zero); irrelevant for timing.
    alpha = jnp.exp(g)
    ea = (b * k) * alpha
    z = w * v
    num_chunks = q.shape[1] // chunk_size
    out = gdn2_recurrence_safe(k, ea, z, alpha, q, chunk_size, num_chunks, bsz, n_heads, d_head, q.dtype)
    return out, jnp.zeros((bsz, n_heads, d_head, d_head), dtype=q.dtype)  # dummy h_final for a uniform (o, h) signature


# ============================================================
# Benchmark harness
# ============================================================
def make_inputs(key, B, L, H, D):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    eps = 1e-6
    q = jax.random.normal(k1, (B, L, H, D), dtype=jnp.float32) * 0.3
    k = jax.random.normal(k2, (B, L, H, D), dtype=jnp.float32) * 0.3
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)
    v = jax.random.normal(k3, (B, L, H, D), dtype=jnp.float32) * 0.3
    g = -jnp.abs(jax.random.normal(k4, (B, L, H, D), dtype=jnp.float32)) * 0.05
    b = jax.random.uniform(k5, (B, L, H, D), minval=0.2, maxval=1.0, dtype=jnp.float32)
    w = jax.random.uniform(k6, (B, L, H, D), minval=0.2, maxval=1.0, dtype=jnp.float32)
    h0 = jax.random.normal(k7, (B, H, D, D), dtype=jnp.float32) * 0.1
    return q, k, v, w, b, g, h0


def time_fn(fn, *args, n_warmup=2, n_iters=5):
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_iters


def _loss_fn(fwd_fn):
    def loss(q, k, v, w, b, g, h0):
        o, h_final = fwd_fn(q, k, v, w, b, g, h0)
        return jnp.sum(o ** 2) + jnp.sum(h_final ** 2)
    return loss


def bench_side(fwd_raw, q, k, v, w, b, g, h0):
    """Returns (t_fwd, t_bwd_only, t_fwd_plus_bwd)."""
    fwd_jit = jax.jit(fwd_raw)
    t_fwd = time_fn(fwd_jit, q, k, v, w, b, g, h0)

    loss = _loss_fn(fwd_raw)

    def bwd_only(q, k, v, w, b, g, h0):
        _, vjp_fn = jax.vjp(loss, q, k, v, w, b, g, h0)
        cot = jnp.array(1.0, dtype=jnp.float32)
        return vjp_fn(cot)

    bwd_jit = jax.jit(bwd_only)
    t_bwd = time_fn(bwd_jit, q, k, v, w, b, g, h0)

    fwdbwd_jit = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5, 6)))
    t_fwdbwd = time_fn(fwdbwd_jit, q, k, v, w, b, g, h0)

    return t_fwd, t_bwd, t_fwdbwd


def bench_one(key, B, L, H, D, chunk_size):
    assert L % chunk_size == 0, f"L={L} must be divisible by chunk_size={chunk_size}"
    q, k, v, w, b, g, h0 = make_inputs(key, B, L, H, D)
    scale = D ** -0.5

    fwd_old_raw = lambda q, k, v, w, b, g, h0: old_associative_scan_forward(
        q, k, v, w, b, g, scale, h0, chunk_size, H, D, B)
    fwd_ref_raw = lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
        q, k, v, g, b, w, scale, chunk_size=chunk_size, h0=h0)
    fwd_pallas_raw = lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
        q, k, v, w, b, g, scale, h0=h0)

    old_times = bench_side(fwd_old_raw, q, k, v, w, b, g, h0)
    ref_times = bench_side(fwd_ref_raw, q, k, v, w, b, g, h0)
    pallas_times = bench_side(fwd_pallas_raw, q, k, v, w, b, g, h0)
    return old_times, ref_times, pallas_times


if __name__ == "__main__":
    configs = [
        dict(B=1, L=1024, H=6, D=128, chunk_size=PALLAS_BT),
        dict(B=4, L=4096, H=6, D=128, chunk_size=PALLAS_BT),
        dict(B=8, L=4096, H=6, D=128, chunk_size=PALLAS_BT),   # train.py exact shape
    ]
    key = jax.random.PRNGKey(6)

    for i, cfg in enumerate(configs):
        subkey = jax.random.fold_in(key, i)
        print(f"\n=== B={cfg['B']} L={cfg['L']} H={cfg['H']} D={cfg['D']} chunk={cfg['chunk_size']} ===")
        try:
            (t_fwd_o, t_bwd_o, t_fb_o), (t_fwd_r, t_bwd_r, t_fb_r), (t_fwd_p, t_bwd_p, t_fb_p) = bench_one(subkey, **cfg)

            print(f"{'stage':>10} {'fwd(ms)':>10} {'bwd(ms)':>10} {'fwd+bwd(ms)':>12}")
            print(f"{'OLD(assoc)':>10} {t_fwd_o*1000:>10.2f} {t_bwd_o*1000:>10.2f} {t_fb_o*1000:>12.2f}")
            print(f"{'JAX_REF':>10} {t_fwd_r*1000:>10.2f} {t_bwd_r*1000:>10.2f} {t_fb_r*1000:>12.2f}")
            print(f"{'PALLAS':>10} {t_fwd_p*1000:>10.2f} {t_bwd_p*1000:>10.2f} {t_fb_p*1000:>12.2f}")

            print(f"\n{'speedup':>18} {'fwd':>8} {'bwd':>8} {'fwd+bwd':>8}")
            print(f"{'OLD -> JAX_REF':>18} {t_fwd_o/t_fwd_r:>7.2f}x {t_bwd_o/t_bwd_r:>7.2f}x {t_fb_o/t_fb_r:>7.2f}x")
            print(f"{'JAX_REF -> PALLAS':>18} {t_fwd_r/t_fwd_p:>7.2f}x {t_bwd_r/t_bwd_p:>7.2f}x {t_fb_r/t_fb_p:>7.2f}x")
            print(f"{'OLD -> PALLAS':>18} {t_fwd_o/t_fwd_p:>7.2f}x {t_bwd_o/t_bwd_p:>7.2f}x {t_fb_o/t_fb_p:>7.2f}x")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
