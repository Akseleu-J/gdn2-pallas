# Changelog

All notable user-facing changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-05

### Fixed

- Renamed `CHANGEOLOG.md` -> `CHANGELOG.md` (referenced correctly by
  `CONTRIBUTING.md` but the file itself had a typo in its name).
- Fixed stale script names in `docs/TESTING_STRATEGY.md`
  (`run_speed.py`/`run_memory.py` -> `run_speed_benchmark.py`/`run_memory_benchmark.py`).

<!-- If 0.1.1 also ships the hybrid-model beta groundwork, add those
     entries here too, e.g.:
### Added
- (beta, opt-in) Hybrid `<name>` module scaffolding -- not yet part of the
  public `atomic_ops` API surface; see `docs/HYBRID_BETA.md` once published.
-->

## [0.1.0] - 2026-09-05

### Added

- Fused GDN-2 forward kernels for TPU v5e in JAX/Pallas:
Kernel A (chunk scores), Kernel B (WY block solve), Kernel C (recompute),
Kernel D (inter-chunk scan).
- Fused backward kernels B1-B5 with a `jax.custom_vjp` trainable wrapper
(`gdn2_pallas_forward_trainable`) that reuses forward residuals instead of
recomputing them.
- Automatic fallback: CPU/GPU or `d_head != 128` dispatches to a checkpointed
pure-JAX chunked-WY reference with identical `wy_eps` damping semantics.
- `KernelConfig` with TPU v5e presets `KAGGLE_SMALL` / `KAGGLE_MEDIUM` /
`KAGGLE_LARGE`, plus `estimate_memory` / `get_recommended_config` helpers.
- Numerical-safety sanitizers at every kernel boundary and a per-stage
diagnostic mode via the `GDN2_FWD_DIAG=1` environment flag.
- Layered correctness suite: CPU smoke tests, multi-seed TPU sweeps,
finite-difference gradient checks, isolated B3-B5 backward-stage tests,
BF16 dtype-contract checks, `wy_eps` damping coverage
(see `docs/TESTING_STRATEGY.md`).
- Speed and memory benchmarks with raw results (`benchmarks/`).

### Known limitations

- The fused forward is currently slower than the pure-JAX WY forward
(~0.62x on TPU v5e-8, forward-only). Training steps are backward-dominated,
so the full cycle is still 2.6-3.9x faster than the best pure-JAX baseline.
- `use_centering=True` is disabled by `KernelConfig` (B4 backward does not
propagate gradients through the centering reference point). See
`KNOWN_LIMITATIONS.md`.

### Performance (TPU v5e-8, train shape B=8, L=4096, 6 heads, d_head=128)

| Metric | FP32 | BF16 |
| --- | --- | --- |
| fwd+bwd vs associative_scan | 27.2x | 13.3x |
| fwd+bwd vs pure-JAX chunked WY | 2.6x | 3.4x |
| Best-case vs associative_scan (all shapes) | 38.8x | 18.7x |

[Unreleased]: https://github.com/Akseleu-J/gdn2-pallas/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Akseleu-J/gdn2-pallas/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Akseleu-J/gdn2-pallas/releases/tag/v0.1.0
