# GDN‑2 Pallas: Performance Benchmarks

All measurements were performed on **Kaggle TPU v5e‑8** (8 cores) with  
`jax==0.11.1`, `jaxlib==0.11.1`, `libtpu==0.0.46`.  
The code is fully reproducible – see the scripts in this folder.

> **Memory benchmarks** are being finalized and will be added in a separate table soon.  
> The numbers below are **steady‑state execution times** (fwdbwd full cycle) averaged over two repeats.

---

## Speed (fwdbwd full cycle, ms)

We compare three implementations:

- **OLD** – associative scan baseline (micro‑batched when necessary).
- **JAX_REF** – chunked WY recurrence written in pure JAX.
- **PALLAS** – our custom Pallas kernels (this library).

### FP32

| Config | OLD (ms) | JAX_REF (ms) | PALLAS (ms) | Speedup vs OLD | Speedup vs JAX_REF |
|--------|----------|--------------|-------------|----------------|---------------------|
| small (B=1,L=1024)   | 143.90 | 22.26 | 5.99 | **24.0×** | 3.72× |
| medium (B=4,L=4096)  | 2655.5 | 268.6 | 88.0 | **30.2×** | 3.05× |
| train (B=8,L=4096)   | 4762.5 | 460.1 | 175.3 | **27.2×** | 2.62× |

### BF16 (recommended for training)

| Config | OLD (ms) | JAX_REF (ms) | PALLAS (ms) | Speedup vs OLD | Speedup vs JAX_REF |
|--------|----------|--------------|-------------|----------------|---------------------|
| small (B=1,L=1024)   | 69.0  | 22.8 | 5.90 | **11.7×** | 3.86× |
| medium (B=4,L=4096)  | 1303.9 | 332.0 | 87.7 | **14.9×** | 3.79× |
| train (B=8,L=4096)   | 2320.4 | 594.9 | 174.4 | **13.3×** | 3.41× |

> **Takeaway:** In BF16 (the dtype used in real training), PALLAS is **~3.4× faster** than the pure‑JAX WY reference and **up to 27× faster** than the legacy associative scan.

---

## Full Results

Detailed per‑stage (fwd / bwd / fwdbwd) tables and raw JSON logs are available in:

- `benchmark_speed_final_averaged.json` – aggregated numbers.
- `benchmark_results_final_averaged.md` – human‑readable full report.

---

## Reproducibility

To reproduce the speed benchmarks:

```bash
python run_speed.py
```

---

This will print the same tables and save JSON summaries.
(For memory benchmarks, use run_memory.py – coming soon.)

Built and benchmarked on Kaggle TPU v5e‑8.
For questions, open an issue or contact the author.
