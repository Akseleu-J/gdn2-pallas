# Roadmap

Status legend:
- **VALIDATED** — measured end-to-end through the public API
  (`gdn2_pallas_forward_trainable` / `gdn2_forward_trainable`), with a
  correctness gate that passed and numbers that are published in
  `benchmarks/raw/` or `README.md`.
- **ISOLATED-ONLY** — a specific kernel or code path has been measured
  or proven correct in isolation (its own test file, its own benchmark
  script), but has NOT yet been measured end-to-end through the full
  pipeline it will ship in. Isolated validation is necessary but not
  sufficient for promotion to a default config.
- **HYPOTHESIS-REJECTED** — an investigated fix direction that measurement
  showed does not work. Kept here so it isn't re-attempted without new
  evidence.
- **OPEN** — a known cost or limitation with no fix in progress.

Nothing in this document is a claim about the current `KAGGLE_*` preset
defaults unless explicitly marked VALIDATED and cross-referenced to a
CHANGELOG entry. Isolated-kernel speedups are real numbers from real
benchmarks, but they are not a substitute for the end-to-end gate.

---

## Now: `use_centering=True` — ISOLATED-ONLY, pending e2e validation

**What has been measured:**

| Item | Status | Evidence |
| --- | --- | --- |
| Kernel A (forward scores) correctness | ISOLATED-ONLY | `test_kernel_a_use_centering_matches_default`, rel_err~2.5e-6 |
| Kernel A (forward scores) speed | ISOLATED-ONLY | `bench_a_centering_speed.py`, 9.23x on train shape |
| Kernel B4 (intra backward) `dgn` gradient correctness | ISOLATED-ONLY | `test_b4_centering_dgc_isolated`, rel_err~3-5e-7 on dq/dk/db/dgc, cross-checked against an independently re-derived `jax.vjp` reference (not the kernel's own code) |
| Kernel B4 (intra backward) speed | ISOLATED-ONLY | `bench_b4_centering_speed.py`: train_shape 62.079ms -> 2.423ms (25.62x); small 5.30x; KAGGLE_SMALL preset 12.18x |
| Full `custom_vjp` pipeline correctness (multi-seed, finite-diff, bf16, KAGGLE_SMALL blocking) | ISOLATED-ONLY (partial) | `test_gdn2_deep_correctness_centering.py` C1-C6 passed with `wy_eps=1e-3`; `finite_diff.b` required 3-seed averaging to suppress a documented WY-solve/FD conditioning artifact (`diag_c1_b_finite_diff.py`), independently confirmed not centering-specific |
| End-to-end fwd/bwd/fwdbwd wall-clock through `gdn2_pallas_forward_trainable` | **NOT YET MEASURED** | pending `run_speed_benchmark.py` run with the `PALLAS_CENTERED` path (patch prepared, not yet executed) |

**Why this is not yet "resolved":** isolated Kernel A / B4 benchmarks
measure those two kernels' own `pallas_call`s in isolation. They do not
by themselves prove that:
- the changed intermediate values (`Aqk`, `Akk` under the centered
  factorization) don't introduce a different cost profile in Kernel C/D
  or in B1/B2/B3 once chained together in the real scan;
  or B3's `dAkk` handoff behave identically once composed rather than
  benchmarked independently.
- there is no compile-time or dispatch regression specific to the
  combination of all Pallas calls together (the section-6 kernel-gap
  diagnostic found this gap to be negligible for the *pre-centering*
  pipeline, but that finding has not been re-run post-centering).

**Gating criteria before promoting `use_centering=True` to the
`KAGGLE_SMALL/MEDIUM/LARGE` defaults:**

1. [ ] Run `benchmarks/run_speed_benchmark.py` with the `PALLAS_CENTERED`
   path (patch: `patches/0001-pallas-centered-e2e-bench.patch`) across all
   `CONFIGS`, both dtypes. Correctness + gradient gates must pass for
   every config (the patch skips timing but not the run for any config
   that fails).
2. [ ] Compare `PALLAS_CENTERED` fwd/bwd/fwdbwd against both `JAX_REF`
   and plain `PALLAS` end-to-end — confirm the isolated-kernel speedups
   (9.23x / 25.62x) survive composition, not just in-isolation.
3. [ ] Re-run the section-6-style kernel-gap diagnostic
   (`sum(isolated per-kernel timing)` vs `full pipeline timing`) for the
   centered path specifically, to rule out a new dispatch/scheduling gap
   introduced by centering's different intermediate shapes.
4. [ ] Peak HBM (`run_memory_benchmark.py`) has not been measured for
   `PALLAS_CENTERED` at all — add before promoting to default, since a
   memory regression would not show up in a speed-only benchmark.
5. [ ] Only after 1-4 pass: flip `KAGGLE_SMALL/MEDIUM/LARGE` to
   `use_centering=True`, lift `NotImplementedError` in
   `KernelConfig.__post_init__`, publish real numbers (not the isolated
   kernel table above) in `README.md`/`CHANGELOG.md`.

**Do not** update `KNOWN_LIMITATIONS.md` sections 1/2 to "RESOLVED" or
change the `KAGGLE_*` preset defaults until item 5 above.

---

## Open, no fix scheduled: Kernel B (WY-solve)

**Status:** OPEN. Confirmed not a tile-size (`mb`) issue via sweep
(32/64/128 gave 44.9-64.0ms, no monotonic relationship) — bottleneck is
the sequential, data-dependent recursive block-forward-substitution in
`_block_solve` (`N_MICRO` sequential steps with data dependency between
them), which has no parallelism to expose to the MXU regardless of block
size.

**Why this now matters more:** once `use_centering=True` clears the
gating criteria above, Kernel B becomes the dominant forward cost by a
wide margin (an estimated ~51ms of ~59.6ms forward, i.e. ~86%, based on
isolated kernel-level extrapolation — itself subject to the same
"not yet e2e-validated" caveat as everything else in this document).

**Candidate directions (none investigated yet):**
- Alternative block-triangular-solve factorization that exposes more
  independent work across micro-blocks (e.g. block-cyclic reduction
  instead of pure forward substitution).
- Investigate whether `N_MICRO` can be reduced by fusing multiple
  micro-block solves into fewer, larger MXU-friendly operations even
  if some redundant computation is introduced.
- `compiled.cost_analysis()` (flops / bytes-accessed) on Kernel B in
  isolation was flagged as "not yet done" as far back as the original
  kernel-gap diagnostic — still not done; would help distinguish
  HBM-bandwidth-bound from genuinely serialization-bound.

No target release. Treat as a standing research item, not a roadmap
milestone with a date.

---

## Completed (VALIDATED, shipped)

- Fused forward + backward Pallas kernels (Kernel A/B/C/D, B1-B5),
  `custom_vjp` trainable wrapper — v0.1.0.
- Kernel B4 fix (`_kernel_b4_body` performance bug) — v0.1.1, ~50%
  backward improvement, measured end-to-end and published in
  `benchmarks/raw/`.

(Nothing from the `use_centering=True` work appears in this section
until the gating criteria above are met.)
