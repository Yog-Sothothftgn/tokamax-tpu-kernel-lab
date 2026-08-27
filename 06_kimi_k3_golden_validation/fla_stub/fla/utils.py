"""Stub for fla.utils -- see package __init__.py for why this exists.

Unlike the other stubs in this package, `tensor_cache` must actually behave
as a valid decorator (not just raise on call): modeling_kimi_linear.py
applies it at MODULE level (`@tensor_cache` on `get_unpad_data`, outside any
class), so decoration happens at import time regardless of whether
`get_unpad_data` itself is ever invoked.
"""


def tensor_cache(fn):
  return fn
