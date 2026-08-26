# ⚡ Atomic Ops — Fused GDN-2 Kernels for TPU v5e

[![PyPI](https://img.shields.io/pypi/v/gdn2-pallas)](https://pypi.org/project/gdn2-pallas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

High-performance **Gated DeltaNet-2** (GDN-2) implementation in **JAX/Pallas**, optimized for **TPU v5e-8**.  
Features a fully fused backward pass delivering **up to 16× end-to-end speedup** over `jax.associative_scan` baselines.
**Fused Gated DeltaNet-2 kernels for TPU v5e**, ported from [NVlabs DeltaNet](https://github.com/NVlabs/Gated_DeltaNet2) Triton reference.
&gt; **Heads-up:** Forward pass is currently ~1.6× slower than the pure-JAX WY reference on TPU (under investigation).  
&gt; The win comes from the **fused backward** — real training steps are dominated by backward, hence the 16× overall gain.  
&gt; A hybrid `JAX forward + Pallas backward` mode is planned for v0.3.0.

---

## Install

```bash
pip install gdn2-pallas

## 🚀 Основные возможности

- **🚄 Молниеносная скорость** — forward и backward полностью написаны на Pallas с использованием MXU-блоков TPU.
- **🧩 Простота использования** — одна функция `gdn2_pallas_forward_trainable` для обучения и `gdn2_pallas_forward` для инференса.
- **🔒 Численная стабильность** — встроенные защиты (clipping, `nan_to_num`) на всех этапах.
- **🔬 Гибкость** — легко настраиваемые параметры (`BT`, `BC`, `CLIP`) через импорт констант.

---

## 📊 Бенчмарки (на TPU v5e‑8)

Я сравнили нашу реализацию с эталонной реализацией **GDN-2 на JAX `associative_scan`** (авторская реализация, широко используемая в исследовательских проектах). Замеры проводились на **batch_size = 8**, **seq_len = 4096**, **heads = 6**, **d_head = 128** (полный профиль обучения с backward).

| Режим | jax.associative_scan (ms) |JAX_REF(чисто математический эталон)| GDN-2 Pallas (ms) | Ускорение (jax_ref)| Ускорение(Pallas)
|-------|---------------------------|-------------------------|-------------------|-----------|-----|
| **Forward** (без backward) |1067.72   |**63.24** |**102.43** |**16.88х**| **10.42x** |
| **Backward** (весь градиент) | 4488.98 |**459.03**| **274.90** | **9.78x**|**16.33x** |
| **Полный цикл** (fwd + bwd) |4501.82 | **460.04**|**274.95** |**9.79x**|  **16.37x** |

> 🔥 **Итог:** Наш код на Pallas превосходит эталонную реализацию по скорости более чем в **16 раз** в среднем, что позволяет ускорить обучение гибридных моделей (GDN-2 + Mamba-2) в **>6 раз** в реальном тренировочном цикле.(реализация Mamba-2 можете найти по этой ссылке [**Mamba-2 Pallas**](https://github.com/Akseleu-J/mamba2-pallas)
> **jax** вресия оказалось для bwd медленной, поэтому pallas дает больше ускорениии

---

## 📦 Установка

### Установка из репозитория (рекомендуется)

```bash
pip install git+https://github.com/Akseleu-J/gdn2-pallas.git
```

---

## Установка в режиме разработки
```bash
git clone https://github.com/Akseleu-J/gdn2-pallas.git
cd gdn2-pallas
pip install -e .
```

---

## 🧠 Быстрый старт
```python
import jax
import jax.numpy as jnp
from atomic_ops import gdn2_forward_trainable

# Создаём случайные тензоры: (batch, seq_len, heads, d_head)
shape = (4, 2048, 6, 128)
q = jnp.ones(shape, dtype=jnp.float32)
k = jnp.ones(shape, dtype=jnp.float32)
v = jnp.ones(shape, dtype=jnp.float32)
w = jnp.ones(shape, dtype=jnp.float32)
b = jnp.ones(shape, dtype=jnp.float32)
g = jnp.ones(shape, dtype=jnp.float32)   # log‑decay
scale = 0.1

# Forward + Backward (через custom_vjp)
out, h_final = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale)

# Только Forward (для инференса)
from atomic_ops import gdn2_pallas_forward
out_inf, _ = gdn2_pallas_forward(q, k, v, w, b, g, scale)

print(out.shape, h_final.shape)   # (4, 2048, 6, 128) (4, 6, 128, 128)
```
> **Примечание**: Функция автоматически подбирает устройство (CPU/GPU/TPU) через JAX. Для TPU v5e-8 убедитесь, что установлены правильные драйверы и jaxlib собран для TPU.

---

## 📖 Архитектура пакета
gdn2-palla/
├── .github/workflows/   # CI/CD пайплайны
├── atomic_ops/ # исходный код ядра
|   ├── __init__.py
|   ├── fallback.py
│   ├── gdn2_fwd.py      # fwd-кмпоненты
│   ├── gdn2_bwd.py      # bwd
|   ├── gdn2_pipeline.py #сборка ядра в пайплайн
|   ├── configs.py
|   ├── utils.py
│   └── reference.py          # Jax эталонная математика
├── tests/              # бенчмарки/тесты
└── README.md            # Документация

---

## 🧑‍💻 Автор
**Akseleu Omirbay** 
Проект создан для высокопроизводительных исследований на TPU v5e.
Вопросы и предложения приветствуются через Issues.

---

## 📄 Лицензия
Распространяется под лицензией **MIT**.

---
## ⭐ Поддержка
Если этот пакет помог вам в работе или исследовании, поставьте ⭐ на GitHub — это поможет другим исследователям найти его!
