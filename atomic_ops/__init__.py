"""
GDN-2 Pallas — production-ready Gated DeltaNet-2 kernels for TPU.
"""
from __future__ import annotations

from .config import KernelConfig, sanitize, sanitize_h0, validate_inputs
from .gdn2_fwd import gdn2_pallas_forward, gdn2_pallas_forward_with_residuals
from .gdn2_pipeline import gdn2_pallas_forward_trainable
from .reference import (
    gdn2_token_serial_reference,
    gdn2_chunked_wy_reference,
)

__version__ = "0.2.0"

__all__ = [
    "KernelConfig",
    "sanitize",
    "sanitize_h0",
    "validate_inputs",
    "gdn2_pallas_forward",
    "gdn2_pallas_forward_with_residuals",
    "gdn2_pallas_forward_trainable",
    "gdn2_token_serial_reference",
    "gdn2_chunked_wy_reference",
]
