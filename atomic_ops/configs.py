from __future__ import annotations
import dataclasses as dc
import jax.lax as lax


@dc.dataclass(frozen=True)
class KernelConfig:
    bt: int = 256
    bc: int = 128
    mb: int = 16
    clip: float = 1e4
    precision: lax.Precision = lax.Precision.HIGHEST

    @property
    def n_sub(self) -> int:
        return self.bt // self.bc

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb


# Пресеты под Kaggle TPU v5e-8
KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4)   # batch 1-2, seq <= 1024
KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4)  # batch 4, seq <= 4096
KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3)   # batch 8, seq 4096 (твой train.py)
