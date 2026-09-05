# Known Limitations

This document lists deliberate restrictions and known gaps of the current
release. Each item includes the reason and the recommended workaround.
Items marked [planned] have a targeted fix in a future release.

## 1. `use_centering=True` is refused by `KernelConfig`

**Status:** intentionally disabled.
**Reason:** the B4 backward kernel (`_kernel_b4_body`) does not propagate the
gradient contribution through the shared centering reference point `gn` into
`dgc`. Constructing a config with `use_centering=True` raises
`NotImplementedError` from `KernelConfig.__post_init__` so that no one can
accidentally train through it via `gdn2_pallas_forward_trainable`.
**Workaround:** forward-only consumers may call the underlying forward kernels
(`build_chunk_scores_pallas`) directly with an unfrozen
`dataclasses.replace(DEFAULT_CONFIG, use_centering=True)` config, and take
responsibility for not differentiating through it.

## 2. Fused forward is slower than the pure-JAX WY forward

**Status:** known performance gap. [planned]
**Numbers (TPU v5e-8, B=8, L=4096):** Pallas fwd 102.08 ms (FP32) / 101.51 ms
(BF16) vs JAX_REF fwd 63.24 ms / 62.38 ms -- about 0.62x.
**Reason:** the fused kernel trades forward speed for a backward that reuses
forward residuals (2.6-3.9x faster than JAX_REF). Training steps are
backward-dominated, so end-to-end is a clear win.
**Workaround:** for inference-only workloads use `gdn2_forward` (dispatches to
the pure-JAX reference off-TPU) or `gdn2_chunked_wy_reference` directly.
A hybrid `JAX forward + Pallas backward` mode is planned for v0.3.0.

## 3. Fused kernels are TPU-only and require `d_head = 128`

The Pallas path assumes TPU MXU tiling. On CPU/GPU, or with `d_head != 128`,
the public API automatically falls back to the pure-JAX chunked-WY reference
(slower, correct). Only Kernel A (`build_chunk_scores_pallas`) and Kernel B4
(`intra_backward_pallas`) accept `interpret=True` and can execute on CPU;
the remaining kernels lower via Mosaic and require a TPU regardless of shape.

## 4. Shape constraints

- `seq_len` must be divisible by `config.bt` (256 by default, 128 for
`KAGGLE_SMALL`).
- `KernelConfig.bt` must equal `2 * config.bc`; vary `mb` for solver
granularity. Other `bt/bc` ratios raise `ValueError` by design (the
top-level WY solve supports only the 2-block split).

## 5. Diagnostics flag is read at import time

`GDN2_FWD_DIAG` is evaluated when `atomic_ops.configs` is first imported.
Set it before starting the Python process.
