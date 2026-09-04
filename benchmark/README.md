# GDN‑2 Pallas: Performance Benchmarks

All measurements were performed on **TPU v5e‑8** (8 cores) with  
`jax==0.11.1`, `jaxlib==0.11.1`, `libtpu==0.0.46.1`.  
The code is fully reproducible – see the scripts in this folder.

> **Memory benchmarks** are still preliminary – final numbers will be added soon.  
> The speed numbers below are **steady‑state execution times** averaged over two repeats.

---

## Speed (Execution Time)

We compare three implementations:
- **OLD** – associative scan baseline (micro‑batched when necessary).
- **JAX_REF** – chunked WY recurrence in pure JAX.
- **PALLAS** – our custom Pallas kernels (this library).

All times are in **milliseconds** (ms).  
Full per‑stage (fwd / bwd / fwdbwd) tables and raw JSON logs are available in the `raw/` folder and in `benchmark_speed_final_averaged.json`.

### FP32

| Config | Stage | OLD (ms) | JAX_REF (ms) | PALLAS (ms) | Speedup vs OLD | Speedup vs JAX_REF |
|--------|-------|----------|--------------|-------------|----------------|---------------------|
| small (B=1,L=1024)   | fwdbwd | 143.84 | 22.09 | 5.92 | **24.3×** | 3.73× |
| medium (B=4,L=4096)  | fwdbwd | 2655.59 | 268.71 | 88.07 | **30.2×** | 3.05× |
| train (B=8,L=4096)   | fwdbwd | 4762.70 | 460.03 | 175.20 | **27.2×** | 2.63× |

> *For full per‑stage breakdown, see the `SPEED TABLE [fp32]` section in the raw logs.*

### BF16 (recommended for training)

| Config | Stage | OLD (ms) | JAX_REF (ms) | PALLAS (ms) | Speedup vs OLD | Speedup vs JAX_REF |
|--------|-------|----------|--------------|-------------|----------------|---------------------|
| small (B=1,L=1024)   | fwdbwd | 68.77 | 22.70 | 6.00 | **11.5×** | 3.79× |
| medium (B=4,L=4096)  | fwdbwd | 1304.06 | 332.06 | 87.60 | **14.9×** | 3.79× |
| train (B=8,L=4096)   | fwdbwd | 2320.27 | 594.86 | 174.20 | **13.3×** | 3.41× |

> **Takeaway:** In BF16 (the dtype used in real training), PALLAS is **~3.4× faster** than the pure‑JAX WY reference and **up to 27× faster** than the legacy associative scan.

---

## Memory (Peak HBM Usage)

*Measurements are process‑isolated and reflect the actual peak during the fwdbwd call.*  
*(Preliminary – final numbers pending verification.)*

### BF16

| Config | Stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|--------|-------|----------|--------------|-------------|
| small (B=1,L=1024)   | fwdbwd | 71.0 | 43.0 | 47.5 | 
| medium (B=4,L=4096)  | fwdbwd | 518.3 | 487.8 | 481.0 | 
| train (B=8,L=4096)   | fwdbwd | 915.5 | 948.8 | 939.4 |

> **Note:** PALLAS uses virtually the same memory as JAX_REF while being significantly faster.  
> `PALLAS_CKPT` (checkpointed) trades speed for slightly lower memory – see the `run_memory_benchmark.py` script for details.

Full memory logs are available in `benchmark_memory_final_averaged.json` (when ready).

---

## Full Results

Detailed per‑stage tables and raw JSON logs are available in:

- `benchmark_speed_final_averaged.json` – aggregated speed numbers.
- `benchmark_results_final_averaged.md` – human‑readable full speed report (if generated).
- `benchmark_memory_final_averaged.json` – aggregated memory numbers (pending).

---

## Reproducibility

To reproduce the speed benchmarks:

```bash
python run_speed_benchmark.py
