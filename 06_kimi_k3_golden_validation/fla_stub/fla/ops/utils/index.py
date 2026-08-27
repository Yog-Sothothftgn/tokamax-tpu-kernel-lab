"""Stub for fla.ops.utils.index -- see package __init__.py for why this exists."""


def prepare_cu_seqlens_from_mask(*args, **kwargs):
  raise NotImplementedError(
      "fla stub: prepare_cu_seqlens_from_mask was actually called -- this "
      "path (attention masking / get_unpad_data) isn't needed for the "
      "MoE-only golden bundle."
  )


def prepare_lens_from_mask(*args, **kwargs):
  raise NotImplementedError(
      "fla stub: prepare_lens_from_mask was actually called -- see "
      "prepare_cu_seqlens_from_mask's note."
  )
