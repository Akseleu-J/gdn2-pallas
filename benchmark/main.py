
from __future__ import annotations
 
import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from functools import partial
from typing import Callable, Optional
 
import jax
import jax.numpy as jnp
 
from atomic_ops.configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from atomic_ops.reference import gdn2_chunked_wy_reference
 
 
# ==========================================================================
# (OLD) associative_scan baseline -- verbatim algorithm, unchanged.
# ==========================================================================
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
    primal_fn = jax.checkpoint(
        lambda k_, ea_, z_, alpha_, q_: _gdn2_recurrence_impl(
            k_, ea_, z_, alpha_, q_, chunk_size, num_chunks, b, n_heads, d_head, dtype
        )
    )
    out, vjp_fn = jax.vjp(primal_fn, k, ea, z, alpha, q)
    return out, vjp_fn
 
 
def _gdn2_recurrence_safe_bwd(chunk_size, num_chunks, b, n_heads, d_head, dtype, vjp_fn, g):
    grads = vjp_fn(g)
    _GRAD_CLIP = 1e3
    return tuple(
        jnp.nan_to_num(jnp.clip(gr, -_GRAD_CLIP, _GRAD_CLIP), nan=0.0, posinf=_GRAD_CLIP, neginf=-_GRAD_CLIP)
        for gr in grads
    )
 
 
gdn2_recurrence_safe.defvjp(_gdn2_recurrence_safe_fwd, _gdn2_recurrence_safe_bwd)
 
 
def old_associative_scan_forward(q, k, v, w, b, g, scale, h0, chunk_size, n_heads, d_head, bsz=None):
    actual_bsz = q.shape[0]
    alpha = jnp.exp(g.astype(jnp.float32))
    ea = ((b * k) * alpha).astype(q.dtype)
    z = (w * v).astype(q.dtype)
    num_chunks = q.shape[1] // chunk_size
    out = gdn2_recurrence_safe(
        k, ea, z, alpha.astype(q.dtype), q, chunk_size, num_chunks, actual_bsz, n_heads, d_head, q.dtype
    )
    out = out.reshape(actual_bsz, out.shape[1], n_heads, d_head)
    return out, jnp.zeros((actual_bsz, n_heads, d_head, d_head), dtype=q.dtype)
 
 
# ==========================================================================
# Config / result data model
# ==========================================================================
@dataclass(frozen=True)
class BenchConfig:
    name: str
    B: int
    L: int
    H: int
    D: int
    kernel_config: KernelConfig
    old_micro_bs: Optional[int] = None
 
 
@dataclass
class StageTiming:
    compile_s: float
    steady_state_s: list
    peak_hbm_bytes: int = -1          # real runtime peak (post-call), see PATCH NOTE 2
    peak_hbm_delta_bytes: int = -1    # peak minus pre-call baseline bytes_in_use
    micro_bs: Optional[int] = None
 
    @property
    def mean(self):
        return statistics.mean(self.steady_state_s)
 
    @property
    def std(self):
        return statistics.pstdev(self.steady_state_s) if len(self.steady_state_s) > 1 else 0.0
 
    @property
    def median(self):
        return statistics.median(self.steady_state_s)
 
    @property
    def min(self):
        return min(self.steady_state_s)
 
    @property
    def max(self):
        return max(self.steady_state_s)
 
    @property
    def cv(self):
        return (self.std / self.mean) if self.mean > 0 else float("nan")
 
 
@dataclass
class PathResult:
    path_name: str
    fwd: StageTiming
    bwd: StageTiming
    fwdbwd: StageTiming
    correctness_ok: bool
    correctness_max_abs_diff: float
    grad_correctness_ok: Optional[bool] = None       # NEW
    grad_correctness_max_rel_err: Optional[float] = None  # NEW
 
 
@dataclass
class ConfigResult:
    config_name: str
    dtype: str                # NEW: "fp32" or "bf16"
    B: int
    L: int
    H: int
    D: int
    n_seeds: int
    n_iters_per_seed: int
    paths: dict = field(default_factory=dict)
 
 
# ==========================================================================
# NEW: environment / version stamping (PATCH NOTE 6)
# ==========================================================================
def collect_env_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
    }
    try:
        import jaxlib
        info["jaxlib_version"] = jaxlib.__version__
    except Exception:
        info["jaxlib_version"] = "unknown"
    try:
        devs = jax.local_devices()
        info["jax_backend"] = jax.default_backend()
        info["device_count"] = len(devs)
        info["device_kind"] = devs[0].device_kind if devs else "unknown"
        info["device_platform"] = devs[0].platform if devs else "unknown"
    except Exception as e:
        info["device_error"] = str(e)
    try:
        import libtpu
        info["libtpu_version"] = getattr(libtpu, "__version__", "unknown")
    except Exception:
        info["libtpu_version"] = "n/a (not importable / not TPU)"
    return info
 
 
def print_env_info(info: dict):
    print("\n" + "=" * 88)
    print("ENVIRONMENT (pin these before treating any number below as final for publishing)")
    print("=" * 88)
    for k, v in info.items():
        print(f"  {k:>20}: {v}")
 
 
# ==========================================================================
# NEW: real runtime peak-HBM measurement (PATCH NOTE 2 + 3)
# ==========================================================================
def _reset_between_measurements():
    """Best-effort flush of transient buffers between path measurements,
    so one path's leftover compiled executables/buffers pollute the next
    path's peak-HBM reading as little as possible. NOT a substitute for
    process-level isolation -- see PATCH NOTE 2/3 docstring at top of file
    for the full caveat."""
    gc.collect()
    jax.clear_caches()
    # touch the device once so the allocator has a chance to reclaim
    jax.block_until_ready(jnp.zeros((1,)))
 
 
def _get_memory_stats():
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        return stats
    except Exception:
        return None
 
 
# --------------------------------------------------------------------
# FIX (found from a real TPU run's log: `peak_hbm(MB)` was IDENTICAL for
# every path/stage within a config -- e.g. 267.8MB for every single row
# of small_B1_L1024). Root cause: `memory_stats()['peak_bytes_in_use']`
# is a MONOTONIC HIGH-WATER-MARK since process start (or since the
# allocator last happened to shrink) -- it is not "the peak during this
# call", it is "the largest bytes_in_use this process has EVER reached".
# Once the first call in a config (OLD fwd, which includes first-time
# compilation) pushes it up, every subsequent call in that config that
# doesn't exceed it just re-reports the same historical maximum forever.
# `gc.collect()`/`jax.clear_caches()` do not reset this counter -- they
# clear Python-side trace caches, not the device allocator's high-water
# mark. There is no public JAX/PJRT API to reset it short of
# `jax.clear_backends()` (which tears down and reinitializes the whole
# runtime -- far too expensive to call between every stage here, and it
# would force recompilation of everything downstream).
#
# FIX: measure LIVE `bytes_in_use` (which rises AND falls, unlike the
# peak counter) by polling it from a background thread WHILE fn(*args)
# is actually running on-device, and report (max_sampled - baseline) as
# the per-call estimate. This is still an estimate -- Python-thread
# sampling can miss very short spikes between polls, and it shares the
# same allocator as everything else in this one process, so residual
# live buffers from a previous stage can still inflate the baseline --
# but unlike the old peak-counter approach, it actually DIFFERS between
# stages/paths, which is the whole point of measuring it. The gold
# standard remains running each (config, path, stage) in an isolated
# subprocess (or profiling via jax.profiler.trace + xplane) -- flagged
# as a TODO at the bottom of this file.
# --------------------------------------------------------------------
import threading
 
 
def _measure_peak_hbm_runtime(fn, args, poll_interval_s: float = 0.002):
    """Runs fn(*args) once (already-warm/compiled) and returns
    (max_live_bytes_in_use_during_call, delta_vs_pre_call_baseline).
 
    Unlike the old `peak_bytes_in_use`-based version, this samples the
    LIVE (rising-and-falling) `bytes_in_use` counter on a background
    thread while the call is in flight, so it actually reflects THIS
    call's footprint rather than a historical maximum shared across the
    whole process. See the FIX comment immediately above this function
    for the full story (found via a real TPU run where every stage in a
    config reported the identical, clearly-wrong peak_hbm number).
    """
    stats_before = _get_memory_stats()
    baseline = stats_before.get("bytes_in_use", -1) if stats_before else -1
    if baseline < 0:
        # memory_stats() unsupported on this backend/device -- degrade
        # gracefully rather than crash the whole benchmark.
        out = fn(*args)
        jax.block_until_ready(out)
        return -1, -1
 
    samples = [baseline]
    stop_flag = threading.Event()
 
    def _poll():
        while not stop_flag.is_set():
            s = _get_memory_stats()
            if s is not None:
                samples.append(s.get("bytes_in_use", baseline))
            stop_flag.wait(poll_interval_s)
 
    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    try:
        out = fn(*args)
        jax.block_until_ready(out)
    finally:
        stop_flag.set()
        poll_thread.join(timeout=1.0)
 
    # one final synchronous sample right after completion, in case the
    # true peak landed between the last poll and block_until_ready
    stats_after = _get_memory_stats()
    if stats_after is not None:
        samples.append(stats_after.get("bytes_in_use", baseline))
 
    peak_live = max(samples)
    delta = peak_live - baseline
    return int(peak_live), int(delta)
 
 
# ==========================================================================
# Timing harness
# ==========================================================================
def _time_repeated(fn, args_per_seed, n_iters_per_seed):
    first_args = args_per_seed[0]
    t0 = time.perf_counter()
    out = fn(*first_args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
 
    steady = []
    for args in args_per_seed:
        for _ in range(n_iters_per_seed):
            t0 = time.perf_counter()
            out = fn(*args)
            jax.block_until_ready(out)
            steady.append(time.perf_counter() - t0)
    return compile_s, steady
 
 
def _loss_fn(fwd_fn):
    def loss(q, k, v, w, b, g, h0):
        o, h_final = fwd_fn(q, k, v, w, b, g, h0)
        return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(h_final.astype(jnp.float32) ** 2)
    return loss
 
 
# --------------------------------------------------------------------
# Micro-batched gradient accumulation (OOM fix for OLD) -- unchanged
# --------------------------------------------------------------------
def make_microbatched_grad_fns(loss_fn, micro_bs):
    def micro_bwd_step(*micro_args):
        _, vjp_fn = jax.vjp(loss_fn, *micro_args)
        cot = jnp.array(1.0, dtype=jnp.float32)
        return vjp_fn(cot)
 
    def micro_fwdbwd_step(*micro_args):
        val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(*micro_args)
        return val, grads
 
    micro_bwd_jit = jax.jit(micro_bwd_step)
    micro_fwdbwd_jit = jax.jit(micro_fwdbwd_step)
 
    def _iter_microbatches(args):
        B = args[0].shape[0]
        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            yield tuple(a[start:end] for a in args)
 
    def bwd_only(*args):
        grad_chunks = None
        for micro_args in _iter_microbatches(args):
            g = micro_bwd_jit(*micro_args)
            if grad_chunks is None:
                grad_chunks = [[gi] for gi in g]
            else:
                for lst, gi in zip(grad_chunks, g):
                    lst.append(gi)
        return tuple(jnp.concatenate(lst, axis=0) for lst in grad_chunks)
 
    def fwdbwd(*args):
        grad_chunks = None
        val_total = 0.0
        for micro_args in _iter_microbatches(args):
            val, g = micro_fwdbwd_jit(*micro_args)
            val_total = val_total + val
            if grad_chunks is None:
                grad_chunks = [[gi] for gi in g]
            else:
                for lst, gi in zip(grad_chunks, g):
                    lst.append(gi)
        grads = tuple(jnp.concatenate(lst, axis=0) for lst in grad_chunks)
        return val_total, grads
 
    return bwd_only, fwdbwd, micro_bwd_jit, micro_fwdbwd_jit
 
 
def bench_path(path_name, fwd_raw, args_per_seed, n_iters_per_seed,
                micro_bs: Optional[int] = None):
    """Returns a PathResult, or None if this path crashed/OOM'd. Now also
    measures REAL runtime peak HBM for fwd, bwd, AND fwd+bwd (PATCH NOTE
    2), and isolates between stage measurements (PATCH NOTE 3)."""
    label = path_name if micro_bs is None else f"{path_name} (micro_bs={micro_bs})"
    print(f"  -- benchmarking path: {label}")
    try:
        fwd_jit = jax.jit(fwd_raw)
 
        compile_fwd, steady_fwd = _time_repeated(fwd_jit, args_per_seed, n_iters_per_seed)
        _reset_between_measurements()
        hbm_fwd, hbm_fwd_delta = _measure_peak_hbm_runtime(fwd_jit, args_per_seed[0])
        print(f"     fwd:     mean={statistics.mean(steady_fwd)*1000:.2f}ms  "
              f"peak_hbm={hbm_fwd/1e6 if hbm_fwd>=0 else float('nan'):.1f}MB "
              f"(delta={hbm_fwd_delta/1e6 if hbm_fwd_delta>=0 else float('nan'):.1f}MB)")
 
        loss = _loss_fn(fwd_raw)
        _reset_between_measurements()
 
        if micro_bs is None:
            def bwd_only(*args):
                _, vjp_fn = jax.vjp(loss, *args)
                cot = jnp.array(1.0, dtype=jnp.float32)
                return vjp_fn(cot)
            bwd_jit = jax.jit(bwd_only)
            compile_bwd, steady_bwd = _time_repeated(bwd_jit, args_per_seed, n_iters_per_seed)
            _reset_between_measurements()
            hbm_bwd, hbm_bwd_delta = _measure_peak_hbm_runtime(bwd_jit, args_per_seed[0])
 
            fwdbwd_fn = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5, 6))
            fwdbwd_jit = jax.jit(fwdbwd_fn)
            compile_fb, steady_fb = _time_repeated(fwdbwd_jit, args_per_seed, n_iters_per_seed)
            _reset_between_measurements()
            hbm_fwdbwd, hbm_fwdbwd_delta = _measure_peak_hbm_runtime(fwdbwd_jit, args_per_seed[0])
        else:
            bwd_only, fwdbwd_fn, micro_bwd_jit, micro_fwdbwd_jit = make_microbatched_grad_fns(loss, micro_bs)
            compile_bwd, steady_bwd = _time_repeated(bwd_only, args_per_seed, n_iters_per_seed)
            _reset_between_measurements()
            micro_args0 = tuple(a[:micro_bs] for a in args_per_seed[0])
            # NOTE: for micro-batched OLD this is still PER-MICRO-BATCH
            # peak (that's genuinely the largest single live allocation at
            # any instant for this path) -- flagged via micro_bs, not
            # comparable 1:1 to full-batch numbers from other paths.
            hbm_bwd, hbm_bwd_delta = _measure_peak_hbm_runtime(micro_bwd_jit, micro_args0)
 
            compile_fb, steady_fb = _time_repeated(fwdbwd_fn, args_per_seed, n_iters_per_seed)
            _reset_between_measurements()
            hbm_fwdbwd, hbm_fwdbwd_delta = _measure_peak_hbm_runtime(micro_fwdbwd_jit, micro_args0)
 
        print(f"     bwd:     mean={statistics.mean(steady_bwd)*1000:.2f}ms  "
              f"peak_hbm={hbm_bwd/1e6 if hbm_bwd>=0 else float('nan'):.1f}MB "
              f"(delta={hbm_bwd_delta/1e6 if hbm_bwd_delta>=0 else float('nan'):.1f}MB)")
        print(f"     fwd+bwd: mean={statistics.mean(steady_fb)*1000:.2f}ms  "
              f"peak_hbm={hbm_fwdbwd/1e6 if hbm_fwdbwd>=0 else float('nan'):.1f}MB "
              f"(delta={hbm_fwdbwd_delta/1e6 if hbm_fwdbwd_delta>=0 else float('nan'):.1f}MB)")
 
        return PathResult(
            path_name=path_name,
            fwd=StageTiming(compile_fwd, steady_fwd, hbm_fwd, hbm_fwd_delta),
            bwd=StageTiming(compile_bwd, steady_bwd, hbm_bwd, hbm_bwd_delta, micro_bs=micro_bs),
            fwdbwd=StageTiming(compile_fb, steady_fb, hbm_fwdbwd, hbm_fwdbwd_delta, micro_bs=micro_bs),
            correctness_ok=True,
            correctness_max_abs_diff=float("nan"),
        )
    except Exception as e:
        print(f"    [FAILED] path {label} raised {type(e).__name__} -- "
              f"skipping this path for this config (other paths still reported). "
              f"Detail: {str(e)[:200]}")
        return None
 
 
# ==========================================================================
# Correctness gates
# ==========================================================================
def _check_correctness(reference_out, candidate_out, tol=1e-2):
    ref_o, ref_h = reference_out
    cand_o, cand_h = candidate_out
    if not (bool(jnp.all(jnp.isfinite(ref_o))) and bool(jnp.all(jnp.isfinite(cand_o)))):
        return False, float("inf")
    diff = float(jnp.max(jnp.abs(ref_o.astype(jnp.float32) - cand_o.astype(jnp.float32))))
    ok = diff < tol
    return ok, diff
 
 
def _check_grad_correctness(fwd_ref, fwd_candidate, args, rel_tol=5e-2):
    """NEW (PATCH NOTE 4). Computes d(loss)/d(inputs) for both the plain-
    JAX reference path and the candidate path at the SAME inputs, and
    reports the max relative error across all 7 gradient tensors. This is
    the thing actually exercised by the bwd/fwd+bwd benchmark stages --
    the previous version never checked it, only the forward output.
    Relative error per-tensor is computed as
        max(|g_cand - g_ref|) / (max(|g_ref|) + eps)
    eps guards near-zero-gradient tensors (e.g. dw for inputs that barely
    affect the loss) from producing meaninglessly huge relative errors.
    """
    loss_ref = _loss_fn(fwd_ref)
    loss_cand = _loss_fn(fwd_candidate)
 
    try:
        grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
        grads_cand = jax.grad(loss_cand, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
    except Exception as e:
        return False, float("inf"), f"grad computation raised {type(e).__name__}: {str(e)[:200]}"
 
    names = ("dq", "dk", "dv", "dw", "db", "dg", "dh0")
    max_rel = 0.0
    per_tensor = {}
    all_finite = True
    for name, gr, gc in zip(names, grads_ref, grads_cand):
        gr32 = gr.astype(jnp.float32)
        gc32 = gc.astype(jnp.float32)
        finite = bool(jnp.all(jnp.isfinite(gr32))) and bool(jnp.all(jnp.isfinite(gc32)))
        all_finite = all_finite and finite
        if not finite:
            per_tensor[name] = float("inf")
            max_rel = float("inf")
            continue
        denom = float(jnp.max(jnp.abs(gr32))) + 1e-6
        rel = float(jnp.max(jnp.abs(gc32 - gr32))) / denom
        per_tensor[name] = rel
        max_rel = max(max_rel, rel)
 
    ok = all_finite and (max_rel < rel_tol)
    detail = ", ".join(f"{k}={v:.2e}" for k, v in per_tensor.items())
    return ok, max_rel, detail
 
 
# ==========================================================================
# Input generation
# ==========================================================================
def make_inputs(key, B, L, H, D, dtype=jnp.float32):
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
 
    # NEW (PATCH NOTE 5): cast the six sequence tensors to the target
    # dtype, matching gdn2_pipeline.py's custom_vjp boundary contract
    # (bf16 forward inputs, fp32 internals). h0 stays fp32 -- production
    # never casts the recurrent state to bf16.
    if dtype != jnp.float32:
        q, k, v, g, b, w = (t.astype(dtype) for t in (q, k, v, g, b, w))
    return q, k, v, w, b, g, h0
 
 
# ==========================================================================
# Main benchmark loop (single repeat, single dtype)
# ==========================================================================
N_SEEDS = 2           # reduced from 3 per request
N_ITERS_PER_SEED = 2  # reduced from 8 per request
 
 
def bench_one_config(bench_cfg: BenchConfig, base_key, dtype, n_seeds=N_SEEDS, n_iters=N_ITERS_PER_SEED):
    B, L, H, D = bench_cfg.B, bench_cfg.L, bench_cfg.H, bench_cfg.D
    kcfg = bench_cfg.kernel_config
    chunk_size = kcfg.bt
    assert L % chunk_size == 0, f"L={L} must be divisible by bt={chunk_size}"
 
    scale = D ** -0.5
    dtype_name = "bf16" if dtype != jnp.float32 else "fp32"
    args_per_seed = [make_inputs(jax.random.fold_in(base_key, s), B, L, H, D, dtype=dtype)
                      for s in range(n_seeds)]
 
    fwd_old = lambda q, k, v, w, b, g, h0: old_associative_scan_forward(
        q, k, v, w, b, g, scale, h0, chunk_size, H, D, bsz=None)
    fwd_ref = lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
        q, k, v, g, b, w, scale, chunk_size=chunk_size, h0=h0, wy_eps=kcfg.wy_eps)
    fwd_pallas = lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
        q, k, v, w, b, g, scale, h0=h0, config=kcfg)
 
    print(f"\n=== {bench_cfg.name} [{dtype_name}]: B={B} L={L} H={H} D={D} "
          f"(bt={kcfg.bt}, bc={kcfg.bc}, mb={kcfg.mb}, wy_eps={kcfg.wy_eps}, "
          f"old_micro_bs={bench_cfg.old_micro_bs}) ===")
 
    print("  forward correctness gate (vs JAX_REF, tol=1e-2, informational only):")
    ref_out = fwd_ref(*args_per_seed[0])
    old_out = fwd_old(*args_per_seed[0])
    pallas_out = fwd_pallas(*args_per_seed[0])
    old_ok, old_diff = _check_correctness(ref_out, old_out)
    pallas_ok, pallas_diff = _check_correctness(ref_out, pallas_out)
    print(f"    OLD    vs JAX_REF: max_abs_diff={old_diff:.4e}  "
          f"(NOTE: OLD ignores h0 entirely -- expected, not a gate)")
    print(f"    PALLAS vs JAX_REF: max_abs_diff={pallas_diff:.4e}  {'OK' if pallas_ok else 'FAILED GATE'}")
 
    print("  gradient correctness gate (NEW -- vs JAX_REF autodiff, rel_tol=5e-2):")
    pallas_grad_ok, pallas_grad_relerr, pallas_grad_detail = _check_grad_correctness(
        fwd_ref, fwd_pallas, args_per_seed[0])
    print(f"    PALLAS grads vs JAX_REF: max_rel_err={pallas_grad_relerr:.4e}  "
          f"{'OK' if pallas_grad_ok else 'FAILED GATE'}  [{pallas_grad_detail}]")
 
    if not pallas_ok:
        print("    [!!] PALLAS failed the FORWARD correctness gate -- SKIPPING timing for this config.")
        return None
    if not pallas_grad_ok:
        print("    [!!] PALLAS failed the GRADIENT correctness gate -- SKIPPING timing for this config "
              "(bwd/fwd+bwd numbers would be timing an incorrect computation).")
        return None
 
    result = ConfigResult(
        config_name=bench_cfg.name, dtype=dtype_name, B=B, L=L, H=H, D=D,
        n_seeds=n_seeds, n_iters_per_seed=n_iters,
    )
 
    _reset_between_measurements()
    old_res = bench_path("OLD (associative_scan)", fwd_old, args_per_seed, n_iters,
                          micro_bs=bench_cfg.old_micro_bs)
    if old_res is not None:
        old_res.correctness_ok, old_res.correctness_max_abs_diff = True, old_diff
 
    _reset_between_measurements()
    ref_res = bench_path("JAX_REF (chunked-WY, plain JAX)", fwd_ref, args_per_seed, n_iters)
    if ref_res is not None:
        ref_res.correctness_ok, ref_res.correctness_max_abs_diff = True, 0.0
 
    _reset_between_measurements()
    pallas_res = bench_path("PALLAS (Atomic_ops)", fwd_pallas, args_per_seed, n_iters)
    if pallas_res is not None:
        pallas_res.correctness_ok, pallas_res.correctness_max_abs_diff = pallas_ok, pallas_diff
        pallas_res.grad_correctness_ok = pallas_grad_ok
        pallas_res.grad_correctness_max_rel_err = pallas_grad_relerr
 
    if ref_res is None or pallas_res is None:
        print("    [!!] JAX_REF or PALLAS crashed/OOM'd -- skipping config.")
        return None
    if old_res is None:
        print("    [NOTE] OLD still crashed/OOM'd even with micro-batching enabled "
              "for this config -- consider lowering old_micro_bs further.")
 
    result.paths = {k: v for k, v in
                     (("OLD", old_res), ("JAX_REF", ref_res), ("PALLAS", pallas_res))
                     if v is not None}
    return result
 
 
CONFIGS = [
    BenchConfig("small_B1_L1024", B=1, L=1024, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=None),
    BenchConfig("medium_B4_L4096", B=4, L=4096, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=None),
    BenchConfig("train_shape_B8_L4096", B=8, L=4096, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=2),
    BenchConfig("kaggle_small_preset_B4_L2048", B=4, L=2048, H=6, D=128, kernel_config=KAGGLE_SMALL,
                old_micro_bs=None),
    BenchConfig("kaggle_large_preset_B8_L4096", B=8, L=4096, H=6, D=128, kernel_config=KAGGLE_LARGE,
                old_micro_bs=2),
]
 
DTYPES = [jnp.float32, jnp.bfloat16]  # PATCH NOTE 5: run every config under both
 
 
# ==========================================================================
# Reporting for a SINGLE repeat (per-repeat debugging/audit)
# ==========================================================================
def _fmt_ms(x):
    return f"{x * 1000:.2f}"
 
 
def _fmt_mb(x):
    return f"{x/1e6:.1f}" if x is not None and x >= 0 else "n/a"
 
 
def print_stdout_report(results, title="REPORT"):
    for r in results:
        if r is None:
            continue
        print(f"\n--- {r.config_name} [{r.dtype}] (B={r.B} L={r.L} H={r.H} D={r.D}, "
              f"n_seeds={r.n_seeds}, n_iters/seed={r.n_iters_per_seed}) ---")
        print(f"{'path':>10} {'stage':>10} {'compile(ms)':>12} {'mean(ms)':>10} "
              f"{'std(ms)':>9} {'peak_hbm(MB)':>13} {'hbm_delta(MB)':>14} {'micro_bs':>9}")
        for path_name, pr in r.paths.items():
            for stage_name, st in (("fwd", pr.fwd), ("bwd", pr.bwd), ("fwd+bwd", pr.fwdbwd)):
                mb_str = str(st.micro_bs) if st.micro_bs else "-"
                print(f"{path_name:>10} {stage_name:>10} {_fmt_ms(st.compile_s):>12} "
                      f"{_fmt_ms(st.mean):>10} {_fmt_ms(st.std):>9} "
                      f"{_fmt_mb(st.peak_hbm_bytes):>13} {_fmt_mb(st.peak_hbm_delta_bytes):>14} {mb_str:>9}")
        pallas = r.paths.get("PALLAS")
        if pallas is not None and pallas.grad_correctness_max_rel_err is not None:
            print(f"  [grad-correctness] PALLAS max_rel_err={pallas.grad_correctness_max_rel_err:.4e} "
                  f"{'OK' if pallas.grad_correctness_ok else 'FAILED'}")
 
        old, ref, pallas = r.paths.get("OLD"), r.paths.get("JAX_REF"), r.paths.get("PALLAS")
        if ref is not None and pallas is not None:
            print(f"\n  {'speedup':>20} {'fwd':>8} {'bwd':>8} {'fwd+bwd':>8}")
            if old is not None:
                print(f"  {'OLD -> JAX_REF':>20} {old.fwd.mean/ref.fwd.mean:>7.2f}x "
                      f"{old.bwd.mean/ref.bwd.mean:>7.2f}x {old.fwdbwd.mean/ref.fwdbwd.mean:>7.2f}x")
            else:
                print(f"  {'OLD -> JAX_REF':>20} n/a (OLD OOM/crashed for this config)")
            print(f"  {'JAX_REF -> PALLAS':>20} {ref.fwd.mean/pallas.fwd.mean:>7.2f}x "
                  f"{ref.bwd.mean/pallas.bwd.mean:>7.2f}x {ref.fwdbwd.mean/pallas.fwdbwd.mean:>7.2f}x")
            if old is not None:
                print(f"  {'OLD -> PALLAS':>20} {old.fwd.mean/pallas.fwd.mean:>7.2f}x "
                      f"{old.bwd.mean/pallas.bwd.mean:>7.2f}x {old.fwdbwd.mean/pallas.fwdbwd.mean:>7.2f}x")
 
 
def dump_json(results, path, env_info):
    serializable = {"env": env_info, "results": []}
    for r in results:
        if r is None:
            continue
        d = dict(config_name=r.config_name, dtype=r.dtype, B=r.B, L=r.L, H=r.H, D=r.D,
                 n_seeds=r.n_seeds, n_iters_per_seed=r.n_iters_per_seed, paths={})
        for name, pr in r.paths.items():
            d["paths"][name] = asdict(pr)
        serializable["results"].append(d)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OUTPUT] Raw results dumped to {path}")
 
 
# ==========================================================================
# Multi-repeat averaging
# ==========================================================================
REPEATS = 2  # reduced from 6 per request
 
 
@dataclass
class AggStage:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    compile_ms: float
    peak_hbm_mb: float
    peak_hbm_delta_mb: float
    micro_bs: Optional[int]
    n_repeats: int
 
    @property
    def cv(self):
        return (self.std_ms / self.mean_ms) if self.mean_ms > 0 else float("nan")
 
 
def _avg(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else float("nan")
 
 
def aggregate_repeats(all_repeat_results, dtype_name):
    """all_repeat_results: list (len REPEATS) of list[ConfigResult|None]
    for ONE dtype, same CONFIGS order each repeat."""
    agg = {}
    n_configs = len(CONFIGS)
    for ci in range(n_configs):
        cfg_name = CONFIGS[ci].name
        per_path = {}
        for path_name in ("OLD", "JAX_REF", "PALLAS"):
            per_stage = {}
            for stage_name in ("fwd", "bwd", "fwdbwd"):
                means, stds, mins, maxs, compiles, hbms, hbm_deltas, mbs = [], [], [], [], [], [], [], None
                for repeat_results in all_repeat_results:
                    r = repeat_results[ci] if ci < len(repeat_results) else None
                    if r is None or r.dtype != dtype_name or path_name not in r.paths:
                        continue
                    pr = r.paths[path_name]
                    st = getattr(pr, stage_name)
                    means.append(st.mean)
                    stds.append(st.std)
                    mins.append(st.min)
                    maxs.append(st.max)
                    compiles.append(st.compile_s)
                    hbms.append(st.peak_hbm_bytes if st.peak_hbm_bytes >= 0 else None)
                    hbm_deltas.append(st.peak_hbm_delta_bytes if st.peak_hbm_delta_bytes >= 0 else None)
                    mbs = st.micro_bs
                if not means:
                    continue
                per_stage[stage_name] = AggStage(
                    mean_ms=_avg(means) * 1000,
                    std_ms=_avg(stds) * 1000,
                    min_ms=min(mins) * 1000,
                    max_ms=max(maxs) * 1000,
                    compile_ms=_avg(compiles) * 1000,
                    peak_hbm_mb=_avg(hbms) / 1e6 if any(h is not None for h in hbms) else float("nan"),
                    peak_hbm_delta_mb=_avg(hbm_deltas) / 1e6 if any(h is not None for h in hbm_deltas) else float("nan"),
                    micro_bs=mbs,
                    n_repeats=len(means),
                )
            if per_stage:
                per_path[path_name] = per_stage
        if per_path:
            agg[cfg_name] = per_path
    return agg
 
 
def print_final_averaged_report(agg, repeats_requested, dtype_name):
    print("\n" + "=" * 96)
    print(f"FINAL AVERAGED REPORT [{dtype_name}] (target {repeats_requested} repeats per config; "
          f"n_repeats column shows how many actually contributed)")
    print("=" * 96)
    for cfg_name, per_path in agg.items():
        print(f"\n--- {cfg_name} [{dtype_name}] ---")
        print(f"{'path':>10} {'stage':>8} {'compile(ms)':>12} {'mean(ms)':>10} "
              f"{'std(ms)':>9} {'cv':>6} {'peak_hbm(MB)':>13} {'hbm_delta(MB)':>14} "
              f"{'micro_bs':>9} {'n_rep':>6}")
        for path_name, per_stage in per_path.items():
            for stage_name in ("fwd", "bwd", "fwdbwd"):
                if stage_name not in per_stage:
                    continue
                s = per_stage[stage_name]
                hbm_str = f"{s.peak_hbm_mb:.1f}" if s.peak_hbm_mb == s.peak_hbm_mb else "n/a"
                hbmd_str = f"{s.peak_hbm_delta_mb:.1f}" if s.peak_hbm_delta_mb == s.peak_hbm_delta_mb else "n/a"
                mb_str = str(s.micro_bs) if s.micro_bs else "-"
                print(f"{path_name:>10} {stage_name:>8} {s.compile_ms:>12.2f} {s.mean_ms:>10.2f} "
                      f"{s.std_ms:>9.2f} {s.cv:>6.3f} {hbm_str:>13} {hbmd_str:>14} "
                      f"{mb_str:>9} {s.n_repeats:>6}")
 
        old = per_path.get("OLD")
        ref = per_path.get("JAX_REF")
        pallas = per_path.get("PALLAS")
        if ref and pallas:
            print(f"\n  {'speedup (avg)':>20} {'fwd':>8} {'bwd':>8} {'fwd+bwd':>8}")
            if old:
                print(f"  {'OLD -> JAX_REF':>20} "
                      f"{old['fwd'].mean_ms/ref['fwd'].mean_ms:>7.2f}x "
                      f"{old['bwd'].mean_ms/ref['bwd'].mean_ms:>7.2f}x "
                      f"{old['fwdbwd'].mean_ms/ref['fwdbwd'].mean_ms:>7.2f}x")
            else:
                print(f"  {'OLD -> JAX_REF':>20} n/a (OLD never succeeded across repeats)")
            print(f"  {'JAX_REF -> PALLAS':>20} "
                  f"{ref['fwd'].mean_ms/pallas['fwd'].mean_ms:>7.2f}x "
                  f"{ref['bwd'].mean_ms/pallas['bwd'].mean_ms:>7.2f}x "
                  f"{ref['fwdbwd'].mean_ms/pallas['fwdbwd'].mean_ms:>7.2f}x")
            if old:
                print(f"  {'OLD -> PALLAS':>20} "
                      f"{old['fwd'].mean_ms/pallas['fwd'].mean_ms:>7.2f}x "
                      f"{old['bwd'].mean_ms/pallas['bwd'].mean_ms:>7.2f}x "
                      f"{old['fwdbwd'].mean_ms/pallas['fwdbwd'].mean_ms:>7.2f}x")
 
 
def dump_final_json(agg_by_dtype, path, env_info, repeats_requested):
    serializable = {
        "env": env_info,
        "repeats_requested": repeats_requested,
        "n_seeds": N_SEEDS,
        "n_iters_per_seed": N_ITERS_PER_SEED,
        "by_dtype": {},
    }
    for dtype_name, agg in agg_by_dtype.items():
        serializable["by_dtype"][dtype_name] = {
            cfg_name: {
                path_name: {stage: asdict(s) for stage, s in per_stage.items()}
                for path_name, per_stage in per_path.items()
            }
            for cfg_name, per_path in agg.items()
        }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OUTPUT] Final averaged results dumped to {path}")
 
 
def write_final_markdown(agg_by_dtype, path, env_info, repeats_requested):
    lines = []
    lines.append("# GDN-2 Atomic_ops benchmark results\n")
    lines.append(f"Averaged over {repeats_requested} full repeats "
                 f"(N_SEEDS={N_SEEDS}, N_ITERS_PER_SEED={N_ITERS_PER_SEED} -- "
                 f"reduced sample counts for a quick smoke-test run; widen "
                 f"before treating these as final publishable numbers, see "
                 f"`cv`/`n_repeats` columns for how noisy each row is).\n")
 
    lines.append("## Environment\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k, v in env_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
 
    lines.append("## Methodology notes\n")
    lines.append("- **peak_hbm(MB)** is a REAL runtime measurement from "
                 "`device.memory_stats()['peak_bytes_in_use']`, sampled "
                 "immediately after each timed call (best-effort isolation "
                 "via `gc.collect()`+`jax.clear_caches()` between "
                 "measurements). It is a monotonic high-water-mark, not a "
                 "perfectly call-isolated peak -- see `hbm_delta(MB)` "
                 "(peak minus the immediately-preceding baseline "
                 "`bytes_in_use`) as a second signal, and treat both as "
                 "estimates, not exact single-call figures. For a fully "
                 "rigorous number, re-run with `jax.profiler.trace` and "
                 "parse the resulting xplane.\n")
    lines.append("- **Gradient correctness gate**: PALLAS gradients "
                 "(d(loss)/d(q,k,v,w,b,g,h0)) are compared against plain-"
                 "JAX autodiff through `gdn2_chunked_wy_reference` at the "
                 "first seed of each config; max relative error per tensor "
                 "is gated at 5e-2. A config failing this gate is excluded "
                 "from timing/speedup entirely (previously only the "
                 "forward *output* was gated, not the gradients that bwd/"
                 "fwd+bwd actually measure).\n")
    lines.append("- **fp32 and bf16 are separate, non-comparable sections** "
                 "below -- bf16 matches the production `custom_vjp` dtype "
                 "contract (bf16 in/out, fp32 internals); fp32 is the "
                 "previous benchmark's dtype. Do not average or directly "
                 "compare timings across the two sections without noting "
                 "the dtype difference explicitly.\n")
    lines.append("- OLD's bwd/fwd+bwd at large B used batch-axis micro-"
                 "batching (`micro_bs` column) to avoid an "
                 "associative_scan-tree OOM; its peak_hbm is per-micro-"
                 "batch, not directly comparable to JAX_REF/PALLAS's "
                 "full-batch peak_hbm.\n")
 
    for dtype_name, agg in agg_by_dtype.items():
        lines.append(f"\n# dtype = {dtype_name}\n")
        for cfg_name, per_path in agg.items():
            lines.append(f"\n## {cfg_name} [{dtype_name}]\n")
            lines.append("| path | stage | compile (ms) | mean (ms) | std (ms) | cv | "
                         "peak HBM (MB) | HBM delta (MB) | micro_bs | n_repeats |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for path_name, per_stage in per_path.items():
                for stage_name in ("fwd", "bwd", "fwdbwd"):
                    if stage_name not in per_stage:
                        continue
                    s = per_stage[stage_name]
                    hbm_str = f"{s.peak_hbm_mb:.1f}" if s.peak_hbm_mb == s.peak_hbm_mb else "n/a"
                    hbmd_str = f"{s.peak_hbm_delta_mb:.1f}" if s.peak_hbm_delta_mb == s.peak_hbm_delta_mb else "n/a"
                    mb_str = str(s.micro_bs) if s.micro_bs else "-"
                    flag = " ⚠️" if s.cv > 0.15 else ""
                    lines.append(f"| {path_name} | {stage_name} | {s.compile_ms:.2f} | "
                                 f"{s.mean_ms:.2f} | {s.std_ms:.2f} | {s.cv:.3f}{flag} | "
                                 f"{hbm_str} | {hbmd_str} | {mb_str} | {s.n_repeats} |")
 
            old = per_path.get("OLD")
            ref = per_path.get("JAX_REF")
            pallas = per_path.get("PALLAS")
            if ref and pallas:
                lines.append("\n**Speedup (mean steady-state time ratio, averaged over repeats):**\n")
                lines.append("| comparison | fwd | bwd | fwd+bwd |")
                lines.append("|---|---|---|---|")
                if old:
                    lines.append(f"| OLD -> JAX_REF | {old['fwd'].mean_ms/ref['fwd'].mean_ms:.2f}x | "
                                 f"{old['bwd'].mean_ms/ref['bwd'].mean_ms:.2f}x | "
                                 f"{old['fwdbwd'].mean_ms/ref['fwdbwd'].mean_ms:.2f}x |")
                else:
                    lines.append("| OLD -> JAX_REF | n/a | | |")
                lines.append(f"| JAX_REF -> PALLAS | {ref['fwd'].mean_ms/pallas['fwd'].mean_ms:.2f}x | "
                             f"{ref['bwd'].mean_ms/pallas['bwd'].mean_ms:.2f}x | "
                             f"{ref['fwdbwd'].mean_ms/pallas['fwdbwd'].mean_ms:.2f}x |")
                if old:
                    lines.append(f"| OLD -> PALLAS | {old['fwd'].mean_ms/pallas['fwd'].mean_ms:.2f}x | "
                                 f"{old['bwd'].mean_ms/pallas['bwd'].mean_ms:.2f}x | "
                                 f"{old['fwdbwd'].mean_ms/pallas['fwdbwd'].mean_ms:.2f}x |")
 
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OUTPUT] Final averaged markdown written to {path}")
 
 

