### 🚀 Speedups vs Baselines (fwdbwd, TPU v5e‑8)

| Config | FP32 OLD→PALLAS | FP32 JAX_REF→PALLAS | BF16 OLD→PALLAS | BF16 JAX_REF→PALLAS |
|--------|----------------:|--------------------:|----------------:|--------------------:|
| `small_B1_L1024` | 24.30× | 3.73× | 11.46× | 3.78× |
| `medium_B4_L4096` | 30.15× | 3.05× | 14.89× | 3.79× |
| `train_shape_B8_L4096` | 27.18× | 2.63× | 13.32× | 3.41× |
| `kaggle_small_preset_B4_L2048` | **38.77×** | **3.87×** | **18.70×** | **3.93×** |
| `kaggle_large_preset_B8_L4096` | 27.16× | 2.62× | 13.33× | 3.42× |

> **Max speedups:**  
> - FP32 OLD→PALLAS: **38.77×** (`kaggle_small`)  
> - FP32 JAX_REF→PALLAS: **3.87×** (`kaggle_small`)  
> - BF16 OLD→PALLAS: **18.70×** (`kaggle_small`)  
> - BF16 JAX_REF→PALLAS: **3.93×** (`kaggle_small`)
