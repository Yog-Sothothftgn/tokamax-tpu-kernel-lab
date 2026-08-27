"""Minimal stub package satisfying modeling_kimi_linear.py's unconditional
`from fla.modules import ...` / `from fla.ops.kda import ...` imports (see
that file's lines 46-53 -- a bare try/except ImportError that re-raises
with an install hint, meaning the REAL fla-core package is required just to
IMPORT the module at all, even though none of the MoE classes this
project's golden-bundle generator uses actually call into it).

Confirmed via direct source read (2026-08-26) that fla-core's real kernels
(FusedRMSNormGated, ShortConvolution, chunk_kda, fused_recurrent_kda) are
used ONLY inside KimiDeltaAttention (the linear-attention layer), which
this project's golden-bundle generator never instantiates -- it only
exercises KimiMoEGate, KimiSparseMoeBlock, KimiBlockSparseMLP, KimiMLP,
KimiRMSNorm, and SituAndMul. Real fla-core also targets NVIDIA/Triton GPUs
specifically (neither this machine nor the v6e TPU VM has one), so
installing the genuine package wouldn't help here regardless of which
machine this runs on.

Every stub below raises NotImplementedError if actually CALLED, so if the
golden-bundle generator ever accidentally exercises attention/KDA code, it
fails loudly instead of silently producing wrong numbers. The one exception
is `tensor_cache`, which is applied as a decorator at MODULE level in
modeling_kimi_linear.py (to `get_unpad_data`) -- decorator application runs
at import time regardless of whether the decorated function is ever
called, so that one stub must behave as a real (identity) decorator, not
just raise.
"""
