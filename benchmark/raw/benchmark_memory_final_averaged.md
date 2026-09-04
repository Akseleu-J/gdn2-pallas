# Memory Benchmarks – Peak HBM (MB)

All numbers are **peak memory usage** (in MB) measured during the actual call, process‑isolated (fork) to avoid interference.  
Measured on **TPU v5e‑8** with `jax==0.11.1`, `jaxlib==0.11.1`.

Only three representative configs are shown; full raw data is available in `raw/`.

---

## FP32

| Config | Stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|--------|-------|----------|--------------|-------------|
| small_B1_L1024 | fwd | 36.361 | 30.993 | 35.185 |
| small_B1_L1024 | bwd | 95.571 | 72.110 | 75.766 |
| small_B1_L1024 | fwdbwd | 95.605 | 72.168 | 75.822 |
| medium_B4_L4096 | fwd | 422.095 | 417.010 | 420.383 |
| medium_B4_L4096 | bwd | 962.204 | 941.402 | 933.817 |
| medium_B4_L4096 | fwdbwd | 962.264 | 941.478 | 933.897 |
| train_shape_B8_L4096 | fwd | 829.708 | 830.193 | 828.515 |
| train_shape_B8_L4096 | bwd | 503.358 | 1857.038 | 1845.268 |
| train_shape_B8_L4096 | fwdbwd | 503.418 | 1857.137 | 1845.344 |

---

## BF16

| Config | Stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|--------|-------|----------|--------------|-------------|
| small_B1_L1024 | fwd | 31.046 | 31.046 | 31.046 |
| small_B1_L1024 | bwd | 70.918 | 42.922 | 47.461 |
| small_B1_L1024 | fwdbwd | 70.975 | 42.979 | 47.518 |
| medium_B4_L4096 | fwd | 460.421 | 460.421 | 460.421 |
| medium_B4_L4096 | bwd | 518.213 | 487.722 | 480.888 |
| medium_B4_L4096 | fwdbwd | 518.279 | 487.797 | 480.966 |
| train_shape_B8_L4096 | fwd | 915.486 | 915.486 | 915.486 |
| train_shape_B8_L4096 | bwd | 285.590 | 948.672 | 939.341 |
| train_shape_B8_L4096 | fwdbwd | 285.656 | 948.771 | 939.418 |

---

*Note: memory values are the mean peak over 3 seeds; standard deviations were negligible.*
