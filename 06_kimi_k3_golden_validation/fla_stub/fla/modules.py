"""Stub for fla.modules -- see package __init__.py for why this exists."""

import torch.nn as nn


class FusedRMSNormGated(nn.Module):
  """Stub -- only used by KimiDeltaAttention, never instantiated by this
  project's golden-bundle generator (MoE path only)."""

  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, *args, **kwargs):
    raise NotImplementedError(
        "fla stub: FusedRMSNormGated was actually called -- this means "
        "something outside the intended MoE-only golden-bundle path was exercised."
    )


class ShortConvolution(nn.Module):
  """Stub -- see FusedRMSNormGated's docstring."""

  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, *args, **kwargs):
    raise NotImplementedError(
        "fla stub: ShortConvolution was actually called -- this means "
        "something outside the intended MoE-only golden-bundle path was exercised."
    )
