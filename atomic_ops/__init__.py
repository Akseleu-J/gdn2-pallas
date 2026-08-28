"""
Atomic Ops — Fused Gated DeltaNet-2 kernels for TPU v5e (Pallas/JAX).
Ported from NVlabs DeltaNet Triton kernels.
"""
from .configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE, DEFAULT_CONFIG
from .utils import is_tpu_available, estimate_memory, get_recommended_config
from .fallback import gdn2_forward, gdn2_forward_trainable
from .gdn2_fwd import gdn2_pallas_forward
from .gdn2_pipeline import gdn2_pallas_forward_trainable
from .reference import gdn2_chunked_wy_reference, gdn2_token_serial_reference

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("gdn2-pallas")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
__all__ = [
    "KernelConfig",
    "KAGGLE_SMALL",
    "KAGGLE_MEDIUM",
    "KAGGLE_LARGE",
    "is_tpu_available",
    "estimate_memory",
    "get_recommended_config",
    "gdn2_forward",
    "gdn2_forward_trainable",
    "gdn2_pallas_forward",
    "gdn2_pallas_forward_trainable",
    "gdn2_chunked_wy_reference",
    "gdn2_token_serial_reference",
]
