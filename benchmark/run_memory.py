from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from functools import partial


SELECTED_CONFIG_NAMES = [
    "small_B1_L1024",           
    "medium_B4_L4096",          
    "train_shape_B8_L4096",     
]

ENABLE_NEIGHBOR_SHAPE_CHECK = False
NEIGHBOR_CONFIG_NAMES = [
    # "neighbor_B8_L3072",
    # "neighbor_B6_L4096",
]

SELECTED_DTYPES = ["fp32", "bf16"]
SELECTED_PATHS = ["OLD", "JAX_REF", "PALLAS"]   # PALLAS_CKPT убран
SELECTED_STAGES = ["fwd", "bwd", "fwdbwd"]
N_SEEDS = 3  # п.5: было 2 -> 3, чтобы иметь запас уверенности в детерминированности

# per-dtype gate tolerances (fp32 -- тугие, bf16 -- ослабленные)
GATE_TOLERANCES = {
    "fp32": dict(fwd_tol=1e-2, grad_rel_tol=5e-2),
    "bf16": dict(fwd_tol=5e-2, grad_rel_tol=1e-1),
}

WORKER_POLL_INTERVAL_S = 0.002
FORK_JOIN_TIMEOUT_S = 900  
FORCE_XLA_PYTHON_CLIENT_PREALLOCATE = None  # None | "true" | "false"
FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION = None  # None | "0.XX"


@dataclass(frozen=True)
class RunSpec:
    config: str
    dtype: str
    path: str
    stage: str
    seed: int


def _build_run_grid():
    config_names = list(SELECTED_CONFIG_NAMES)
    if ENABLE_NEIGHBOR_SHAPE_CHECK:
        config_names = config_names + NEIGHBOR_CONFIG_NAMES
    grid = []
    for config in config_names:
        for dtype in SELECTED_DTYPES:
            for path in SELECTED_PATHS:
                for stage in SELECTED_STAGES:
                    for seed in range(N_SEEDS):
                        grid.append(RunSpec(config, dtype, path, stage, seed))
    return grid


RUN_GRID = _build_run_grid()


CONFIGS = {
    "small_B1_L1024": dict(B=1, L=1024, H=6, D=128, kernel_config="KAGGLE_MEDIUM", old_micro_bs=None),
    "medium_B4_L4096": dict(B=4, L=4096, H=6, D=128, kernel_config="KAGGLE_MEDIUM", old_micro_bs=None),
    "train_shape_B8_L4096": dict(B=8, L=4096, H=6, D=128, kernel_config="KAGGLE_MEDIUM", old_micro_bs=2),
    "kaggle_small_preset_B4_L2048": dict(B=4, L=2048, H=6, D=128, kernel_config="KAGGLE_SMALL", old_micro_bs=None),
    "kaggle_large_preset_B8_L4096": dict(B=8, L=4096, H=6, D=128, kernel_config="KAGGLE_LARGE", old_micro_bs=2),
    "neighbor_B8_L3072": dict(B=8, L=3072, H=6, D=128, kernel_config="KAGGLE_MEDIUM", old_micro_bs=2),
    "neighbor_B6_L4096": dict(B=6, L=4096, H=6, D=128, kernel_config="KAGGLE_MEDIUM", old_micro_bs=2),
}


def _set_and_log_memory_env():
    import os
    if FORCE_XLA_PYTHON_CLIENT_PREALLOCATE is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = FORCE_XLA_PYTHON_CLIENT_PREALLOCATE
    if FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION
    return dict(
        XLA_PYTHON_CLIENT_PREALLOCATE=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset, JAX default=true>"),
        XLA_PYTHON_CLIENT_MEM_FRACTION=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "<unset, JAX default=0.75>"),
        XLA_PYTHON_CLIENT_ALLOCATOR=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "<unset>"),
    )


def _child_main(run_idx: int, result_queue: "mp.Queue"):
    mem_env = _set_and_log_memory_env()

    import os
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/kaggle/working/.jax_cache")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    from atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
    from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
    from atomic_ops.reference import gdn2_chunked_wy_reference

    KERNEL_CONFIGS = {"KAGGLE_SMALL": KAGGLE_SMALL, "KAGGLE_MEDIUM": KAGGLE_MEDIUM, "KAGGLE_LARGE": KAGGLE_LARGE}

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
        _, out_chunks = jax.lax.scan(_chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch))
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
        if dtype != jnp.float32:
            q, k, v, g, b, w = (t.astype(dtype) for t in (q, k, v, g, b, w))
        return q, k, v, w, b, g, h0

    def build_fn(path_name, H, D, kcfg, scale):
        chunk_size = kcfg.bt
        if path_name == "OLD":
            return lambda q, k, v, w, b, g, h0: old_associative_scan_forward(
                q, k, v, w, b, g, scale, h0, chunk_size, H, D, bsz=None)
        if path_name == "JAX_REF":
            return lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
                q, k, v, g, b, w, scale, chunk_size=chunk_size, h0=h0, wy_eps=kcfg.wy_eps)
        if path_name == "PALLAS":
            return lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
                q, k, v, w, b, g, scale, h0=h0, config=kcfg)
        raise ValueError(f"unknown path {path_name}")

    def _loss_fn(fwd_fn):
        def loss(q, k, v, w, b, g, h0):
            o, h_final = fwd_fn(q, k, v, w, b, g, h0)
            return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(h_final.astype(jnp.float32) ** 2)
        return loss

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
        return None, None, micro_bwd_jit, micro_fwdbwd_jit

    def _get_memory_stats():
        try:
            return jax.local_devices()[0].memory_stats()
        except Exception:
            return None

    def _sample_during_call(fn, args, poll_interval_s=WORKER_POLL_INTERVAL_S):
        stats_before = _get_memory_stats()
        baseline = stats_before.get("bytes_in_use", -1) if stats_before else -1
        samples = [baseline] if baseline >= 0 else []
        stop_flag = threading.Event()

        def _poll():
            while not stop_flag.is_set():
                s = _get_memory_stats()
                if s is not None:
                    samples.append(s.get("bytes_in_use", baseline))
                stop_flag.wait(poll_interval_s)

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        try:
            out = fn(*args)
            jax.block_until_ready(out)
        finally:
            stop_flag.set()
            t.join(timeout=1.0)

        stats_after = _get_memory_stats()
        if stats_after is not None:
            samples.append(stats_after.get("bytes_in_use", baseline))
        return out, samples, baseline, stats_after

    spec = RUN_GRID[run_idx]
    result = dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype, path=spec.path,
                  stage=spec.stage, seed=spec.seed, ok=False, error=None,
                  mem_env=mem_env)
    try:
        cfg = CONFIGS[spec.config]
        kcfg = KERNEL_CONFIGS[cfg["kernel_config"]]
        B, L, H, D = cfg["B"], cfg["L"], cfg["H"], cfg["D"]
        micro_bs = cfg["old_micro_bs"]
        scale = D ** -0.5
        dtype = jnp.bfloat16 if spec.dtype == "bf16" else jnp.float32

        key = jax.random.fold_in(jax.random.PRNGKey(6), spec.seed)

        use_micro = micro_bs is not None and spec.path == "OLD" and spec.stage in ("bwd", "fwdbwd")
        inputs = make_inputs(key, micro_bs if use_micro else B, L, H, D, dtype=dtype)
        fwd_fn = build_fn(spec.path, H, D, kcfg, scale)

        loss_fn = _loss_fn(fwd_fn)

        if spec.stage == "fwd":
            call_fn = jax.jit(fwd_fn)
            call_args = inputs
        elif spec.stage == "bwd":
            if use_micro:
                _, _, micro_bwd_jit, _ = make_microbatched_grad_fns(loss_fn, micro_bs)
                call_fn = micro_bwd_jit
                call_args = inputs
            else:
                def bwd_only(*a):
                    _, vjp_fn = jax.vjp(loss_fn, *a)
                    return vjp_fn(jnp.array(1.0, dtype=jnp.float32))
                call_fn = jax.jit(bwd_only)
                call_args = inputs
        elif spec.stage == "fwdbwd":
            if use_micro:
                _, _, _, micro_fwdbwd_jit = make_microbatched_grad_fns(loss_fn, micro_bs)
                call_fn = micro_fwdbwd_jit
                call_args = inputs
            else:
                call_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6)))
                call_args = inputs
        else:
            raise ValueError(spec.stage)

        warm_out = call_fn(*call_args)
        jax.block_until_ready(warm_out)
        gc.collect()

        settle_stats = _get_memory_stats()
        settle_peak = settle_stats.get("peak_bytes_in_use", -1) if settle_stats else -1

        _, samples, _, stats_after = _sample_during_call(call_fn, call_args)

        finite_samples = [s for s in samples if s is not None and s >= 0]

        peak_bytes_in_use_process = stats_after.get("peak_bytes_in_use", -1) if stats_after else -1
        bytes_limit = stats_after.get("bytes_limit", -1) if stats_after else -1
        largest_free_block_bytes = stats_after.get("largest_free_block_bytes", -1) if stats_after else -1
        num_allocs = stats_after.get("num_allocs", -1) if stats_after else -1

        steady_state_confirmed = (
            settle_peak == peak_bytes_in_use_process
            if (settle_peak >= 0 and peak_bytes_in_use_process >= 0) else None
        )

        near_pool_limit = (
            bytes_limit > 0 and peak_bytes_in_use_process > 0
            and (peak_bytes_in_use_process / bytes_limit) > 0.95
        )

        result.update(
            ok=True,
            peak_bytes_in_use_process_raw=peak_bytes_in_use_process,
            settle_peak_bytes_raw=settle_peak,
            steady_state_confirmed=steady_state_confirmed,
            bytes_limit_raw=bytes_limit,
            largest_free_block_bytes_raw=largest_free_block_bytes,
            num_allocs=num_allocs,
            near_pool_limit=near_pool_limit,
            min_bytes_in_use_during_call=min(finite_samples) if finite_samples else -1,
            max_bytes_in_use_during_call=max(finite_samples) if finite_samples else -1,
            n_samples=len(finite_samples),
            micro_bs=micro_bs if (spec.path == "OLD") else None,
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    result_queue.put(result)


def _child_gate_check(config_name: str, dtype_name: str, path_name: str, result_queue: "mp.Queue"):
    _set_and_log_memory_env()
    import jax
    import jax.numpy as jnp
    from atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
    from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
    from atomic_ops.reference import gdn2_chunked_wy_reference

    KERNEL_CONFIGS = {"KAGGLE_SMALL": KAGGLE_SMALL, "KAGGLE_MEDIUM": KAGGLE_MEDIUM, "KAGGLE_LARGE": KAGGLE_LARGE}
    tol = GATE_TOLERANCES[dtype_name]
    fwd_tol, grad_rel_tol = tol["fwd_tol"], tol["grad_rel_tol"]
    result = dict(config=config_name, dtype=dtype_name, path=path_name, ok=False, error=None)
    try:
        cfg = CONFIGS[config_name]
        kcfg = KERNEL_CONFIGS[cfg["kernel_config"]]
        B, L, H, D = cfg["B"], cfg["L"], cfg["H"], cfg["D"]
        scale = D ** -0.5
        dtype = jnp.bfloat16 if dtype_name == "bf16" else jnp.float32
        key = jax.random.fold_in(jax.random.PRNGKey(6), 0)

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
            if dtype != jnp.float32:
                q, k, v, g, b, w = (t.astype(dtype) for t in (q, k, v, g, b, w))
            return q, k, v, w, b, g, h0

        inputs = make_inputs(key, B, L, H, D, dtype=dtype)

        def build_ref(H, D, kcfg, scale):
            return lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
                q, k, v, g, b, w, scale, chunk_size=kcfg.bt, h0=h0, wy_eps=kcfg.wy_eps)

        if path_name == "JAX_REF":
            result.update(ok=True, note="JAX_REF is the reference itself -- no gate needed.")
            result_queue.put(result)
            return
        if path_name == "OLD":
            result.update(ok=True, note="OLD ignores h0 by design -- correctness vs JAX_REF not meaningful.")
            result_queue.put(result)
            return

        if path_name == "PALLAS":
            ref_fn = build_ref(H, D, kcfg, scale)
            cand_fn = lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
                q, k, v, w, b, g, scale, h0=h0, config=kcfg)

            def _loss(fn):
                def loss(q, k, v, w, b, g, h0):
                    o, hf = fn(q, k, v, w, b, g, h0)
                    return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(hf.astype(jnp.float32) ** 2)
                return loss

            ref_out = ref_fn(*inputs)
            cand_out = cand_fn(*inputs)
            ref_o, ref_h = ref_out
            cand_o, cand_h = cand_out
            fwd_diff = float(jnp.max(jnp.abs(ref_o.astype(jnp.float32) - cand_o.astype(jnp.float32))))
            fwd_ok = fwd_diff < fwd_tol and bool(jnp.all(jnp.isfinite(cand_o)))

            grads_ref = jax.grad(_loss(ref_fn), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
            grads_cand = jax.grad(_loss(cand_fn), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
            max_rel = 0.0
            for gr, gcd in zip(grads_ref, grads_cand):
                gr32, gcd32 = gr.astype(jnp.float32), gcd.astype(jnp.float32)
                if not (bool(jnp.all(jnp.isfinite(gr32))) and bool(jnp.all(jnp.isfinite(gcd32)))):
                    max_rel = float("inf")
                    continue
                denom = float(jnp.max(jnp.abs(gr32))) + 1e-6
                max_rel = max(max_rel, float(jnp.max(jnp.abs(gcd32 - gr32))) / denom)
            grad_ok = max_rel < grad_rel_tol

            result.update(ok=True, fwd_correctness_ok=fwd_ok, fwd_correctness_max_abs_diff=fwd_diff,
                          grad_correctness_ok=grad_ok, grad_correctness_max_rel_err=max_rel)
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    result_queue.put(result)


def run_gate_check(config_name, dtype_name, path_name):
    ctx = mp.get_context("fork")
    qres = ctx.Queue()
    p = ctx.Process(target=_child_gate_check, args=(config_name, dtype_name, path_name, qres))
    p.start()
    try:
        r = qres.get(timeout=FORK_JOIN_TIMEOUT_S)
    except Exception:
        r = dict(config=config_name, dtype=dtype_name, path=path_name, ok=False,
                  error="gate-check process produced no result")
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join()
    return r


def run_isolated(run_idx: int) -> dict:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_main, args=(run_idx, q))
    p.start()
    got = None
    try:
        got = q.get(timeout=FORK_JOIN_TIMEOUT_S)
    except Exception:
        pass
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join()
    if got is None:
        spec = RUN_GRID[run_idx]
        return dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype, path=spec.path,
                    stage=spec.stage, seed=spec.seed, ok=False,
                    error=f"child process produced no result (exitcode={p.exitcode})")
    return got


def run_all():
    total = len(RUN_GRID)
    print("=" * 96)
    print("MEMORY-ONLY BENCHMARK v2 fp32+bf16 (fork-isolated, no files, atomic_ops) -- "
          f"{total} isolated forks (neighbor-shape check: {ENABLE_NEIGHBOR_SHAPE_CHECK})")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 96)

    raw_results = []
    gate_results = []

    seen_gate_keys = set()
    for spec in RUN_GRID:
        gkey = (spec.config, spec.dtype, spec.path)
        if gkey in seen_gate_keys:
            continue
        seen_gate_keys.add(gkey)
        print(f"\n[GATE] {spec.config} [{spec.dtype}] {spec.path} ...")
        g = run_gate_check(spec.config, spec.dtype, spec.path)
        gate_results.append(g)
        if g.get("ok"):
            if "grad_correctness_max_rel_err" in g:
                print(f"    grad max_rel_err={g['grad_correctness_max_rel_err']:.4e} "
                      f"{'OK' if g.get('grad_correctness_ok') else 'FAILED'}")
            else:
                print(f"    {g.get('note', 'gate skipped/not applicable')}")
        else:
            print(f"    [GATE ERROR] {g.get('error')}")

    failed_gate_keys = {
        (g["config"], g["dtype"], g["path"])
        for g in gate_results
        if not g.get("ok") or (g.get("grad_correctness_ok") is False)
    }

    for run_idx in range(total):
        spec = RUN_GRID[run_idx]
        gkey = (spec.config, spec.dtype, spec.path)
        label = f"[{spec.config}][{spec.dtype}][{spec.path}][{spec.stage}][seed={spec.seed}]"
        print(f"\n[{run_idx + 1}/{total}] >>> {label}")
        if gkey in failed_gate_keys:
            print("    [SKIPPED] failed correctness/gradient gate for this (config,dtype,path).")
            raw_results.append(dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype,
                                     path=spec.path, stage=spec.stage, seed=spec.seed,
                                     ok=False, error="skipped: failed correctness/gradient gate"))
            continue
        r = run_isolated(run_idx)
        raw_results.append(r)
        if r.get("ok"):
            peak_raw = r.get("peak_bytes_in_use_process_raw", -1)
            limit_raw = r.get("bytes_limit_raw", -1)
            flag = " [NEAR POOL LIMIT!]" if r.get("near_pool_limit") else ""
            steady = r.get("steady_state_confirmed")
            steady_flag = "" if steady is True else (" [STEADY-STATE NOT CONFIRMED -- possible compile-spike contamination!]" if steady is False else "")
            ratio = f"{peak_raw/limit_raw:.3f}" if limit_raw > 0 else "n/a"
            print(f"    [OK] peak_raw={peak_raw} bytes ({peak_raw/1e6:.3f}MB) "
                  f"bytes_limit={limit_raw} ratio={ratio}{flag}{steady_flag}")
            if run_idx == 0:
                print(f"    mem_env={r.get('mem_env')}")
        else:
            print(f"    [FAILED/GATED] {r.get('error')}")

    grouped = defaultdict(list)
    for r in raw_results:
        if r.get("ok"):
            grouped[(r["config"], r["dtype"], r["path"], r["stage"])].append(r)

    agg = defaultdict(lambda: defaultdict(dict))
    for (config, dtype, path, stage), runs in grouped.items():
        # RAW bytes -- никакого округления до агрегации.
        peaks_raw = [r["peak_bytes_in_use_process_raw"] for r in runs
                     if r.get("peak_bytes_in_use_process_raw", -1) >= 0]
        near_limit_any = any(r.get("near_pool_limit") for r in runs)
        agg[(config, dtype)][path][stage] = dict(
            n_seeds_ok=len(runs),
            peak_min_bytes=min(peaks_raw) if peaks_raw else None,
            peak_max_bytes=max(peaks_raw) if peaks_raw else None,
            peak_mean_bytes=(sum(peaks_raw) / len(peaks_raw)) if peaks_raw else None,
            near_pool_limit_any=near_limit_any,
        )

    gate_summary = {}
    for r in raw_results:
        if r.get("path") == "PALLAS" and "grad_correctness_ok" in r:
            gate_summary.setdefault((r["config"], r["dtype"], r["path"]), r)

    failed = [r for r in raw_results if not r.get("ok")]

    print("\n" + "=" * 96)
    print("FINAL REPORT MEMORY-ONLY v2 (fork-isolated, RAW bytes, atomic_ops)")
    print("=" * 96)

    print("\n-- gradient correctness gate (vs JAX_REF autodiff, per-dtype tolerances) --")
    for (config, dtype, path), r in sorted(gate_summary.items()):
        relerr = r.get("grad_correctness_max_rel_err")
        relerr_str = f"{relerr:.2e}" if relerr is not None else "n/a"
        print(f"  {config:>30} [{dtype}] {path:>12}  max_rel_err={relerr_str:>10}  "
              f"{'OK' if r.get('grad_correctness_ok') else 'FAILED'}")

    for (config, dtype), per_path in sorted(agg.items()):
        print(f"\n-- {config} [{dtype}] --")
        print(f"  {'path':>10} {'stage':>8} {'peak min(bytes)':>17} {'peak max(bytes)':>17} "
              f"{'peak mean(MB)':>14} {'n_ok':>5} {'pool_limit?':>12}")
        for path_name, per_stage in per_path.items():
            for stage_name in ("fwd", "bwd", "fwdbwd"):
                if stage_name not in per_stage:
                    continue
                s = per_stage[stage_name]
                def fmtb(x):
                    return f"{x}" if x is not None else "n/a"
                def fmtmb(x):
                    return f"{x/1e6:.3f}" if x is not None else "n/a"
                print(f"  {path_name:>10} {stage_name:>8} {fmtb(s['peak_min_bytes']):>17} "
                      f"{fmtb(s['peak_max_bytes']):>17} {fmtmb(s['peak_mean_bytes']):>14} "
                      f"{s['n_seeds_ok']:>5} {str(s['near_pool_limit_any']):>12}")

    print("\n-- floor-hypothesis check: разброс путей внутри (config,dtype,stage) --")
    for (config, dtype), per_path in sorted(agg.items()):
        for stage_name in ("fwd", "bwd", "fwdbwd"):
            vals = []
            for path_name in SELECTED_PATHS:
                s = per_path.get(path_name, {}).get(stage_name)
                if s and s.get("peak_mean_bytes") is not None:
                    vals.append((path_name, s["peak_mean_bytes"]))
            if len(vals) < 2:
                continue
            nums = [v for _, v in vals]
            spread_pct = (max(nums) - min(nums)) / max(nums) * 100 if max(nums) > 0 else 0.0
            identical = len(set(nums)) == 1
            print(f"  {config} [{dtype}] {stage_name}: {vals} "
                  f"spread={spread_pct:.4f}% identical_bitwise={identical}")

    if failed:
        print(f"\n-- failed / gated-out runs ({len(failed)}) --")
        for r in failed:
            print(f"  {r.get('config')} [{r.get('dtype')}] {r.get('path')} {r.get('stage')} "
                  f"seed={r.get('seed')}: {str(r.get('error'))[:150]}")

    unsteady = [r for r in raw_results if r.get("ok") and r.get("steady_state_confirmed") is False]
    if unsteady:
        print(f"\n-- WARNING: {len(unsteady)} runs did NOT confirm steady-state "
              f"(measured peak > settle-after-warmup peak -- possible compile-spike "
              f"or genuine multi-call growth, needs manual check) --")
        for r in unsteady:
            print(f"  {r.get('config')} [{r.get('dtype')}] {r.get('path')} {r.get('stage')} "
                  f"seed={r.get('seed')}: settle={r.get('settle_peak_bytes_raw')} "
                  f"measured={r.get('peak_bytes_in_use_process_raw')}")

    print("\n" + "=" * 96)
    print(f"[SUMMARY] {len(raw_results) - len(failed)}/{len(raw_results)} isolated runs succeeded.")
    print("=" * 96)

    for dtype_name in SELECTED_DTYPES:
        agg_for_dtype = {k: v for k, v in agg.items() if k[1] == dtype_name}
        print_publish_table(agg_for_dtype, dtype_name)

    return raw_results, agg, gate_summary


def print_publish_table(agg, dtype_name):
    print("\n" + "=" * 96)
    print(f"PUBLISH TABLE -- peak HBM (MB, из raw bytes) ONLY, {dtype_name}, process-isolated (fork)")
    print("=" * 96)
    print("\n| config | stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |")
    print("|---|---|---|---|---|")
    for (config, dtype), per_path in sorted(agg.items()):
        old_p = per_path.get("OLD", {})
        ref_p = per_path.get("JAX_REF", {})
        pal_p = per_path.get("PALLAS", {})
        for stage in ("fwd", "bwd", "fwdbwd"):
            old_b = old_p.get(stage, {}).get("peak_mean_bytes")
            ref_b = ref_p.get(stage, {}).get("peak_mean_bytes")
            pal_b = pal_p.get(stage, {}).get("peak_mean_bytes")

            def fmt(x):
                return f"{x/1e6:.3f}" if x is not None else "n/a"

            print(f"| {config} | {stage} | {fmt(old_b)} | {fmt(ref_b)} | {fmt(pal_b)} |")


if __name__ == "__main__":
    raw_results, agg, gate_summary = run_all()
