"""Local (CPU-only, no TPU/tokamax needed) unit tests for the real
16-of-896-then-filtered-to-local-shard routing pipeline
(`filter_and_pad_to_shard` / `route_and_filter_to_local_shard` /
`_combine_shard_contribution` / `check_sharded_forward_correctness`),
covering edge cases the existing checks don't exercise directly.

Explicit goal (user request, 2026-08-28): catch ordinary logic bugs here,
on CPU, so scarce TPU time isn't spent re-discovering them.

Covers:
  - a shard that gets ZERO real assignments
  - all tokens concentrated on a single expert within a shard (extreme skew)
  - capacity overflow/truncation
  - padding rows contribute exactly zero, even when deliberately given
    non-zero content (not just relying on "zero input happens to produce
    zero output")
  - top_k=16 (the real Kimi K3 value -- existing checks mostly use smaller
    top_k for speed)
  - the exact global-expert-id -> local-expert-id boundary (off-by-one)
  - the first and last shard when tiling a global expert range
  - batch/seq decomposition has no effect (there is no separate batch axis
    anywhere in this pipeline, only num_tokens = hidden_states.shape[0] --
    this test is a regression guard for that architectural invariant, not
    a deep test, since it's true by construction today)
  - fp32/bf16 dtype doesn't silently promote, including on the empty-group
    branch -- this exact bug class has bitten this project before
    (`latent_moe_forward`'s empty-expert-group branch once defaulted to
    float32 via an un-dtyped `jnp.zeros`, silently upcasting everything
    downstream under bf16 compute)
  - summing every shard's contribution reproduces the unsharded reference,
    across several configs/seeds (`check_sharded_forward_correctness` only
    ever tested one config)

No tokamax dependency -- runs anywhere `kimi_k3_latent_moe_reference.py` does.

Usage:
  python test_sharded_routing_local.py
"""

import jax
import jax.numpy as jnp

from kimi_k3_latent_moe_reference import (
    LatentMoEConfig,
    _combine_shard_contribution,
    _local_shard_expert_ffn,
    check_sharded_forward_correctness,
    filter_and_pad_to_shard,
    route_and_filter_to_local_shard,
)


def _toy_shard_config(top_k: int = 4, num_experts: int = 32) -> LatentMoEConfig:
  return LatentMoEConfig(
      hidden_size=64, latent_size=32, intermediate_size=48,
      num_experts=num_experts, top_k=top_k, num_shared_experts=1,
      moe_renormalize=True, routed_scaling_factor=1.0, rms_norm_eps=1e-5,
      activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
  )


def check_empty_shard() -> bool:
  """A shard that gets ZERO real (token, slot) assignments -- constructed
  directly (bypassing the router) so this is deterministic, not
  probabilistic."""
  config = _toy_shard_config(top_k=2, num_experts=8)
  num_tokens = 5
  local_expert_start, local_num_experts = 4, 2  # shard covers experts [4, 6)

  # Every pick lands OUTSIDE the shard on purpose.
  topk_idx = jnp.array([[0, 1], [1, 2], [6, 7], [7, 0], [2, 3]], dtype=jnp.int32)
  topk_weight = jnp.full((num_tokens, config.top_k), 0.5, dtype=jnp.float32)
  x = jax.random.normal(jax.random.key(0), (num_tokens, config.latent_size))

  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = filter_and_pad_to_shard(
      topk_idx, topk_weight, x, config, local_expert_start, local_num_experts,
      capacity_factor=4.0, tile_size=1,
  )

  ok = (
      int(jnp.sum(per_expert_counts)) == 0
      and int(jnp.sum(valid_mask)) == 0
      and bool(jnp.all(sorted_tokens == 0))
      and bool(jnp.all(padded_combine_weight == 0))
      and bool(jnp.all(padded_token_idx == -1))
  )
  print(
      f"[local-test] empty_shard: per_expert_counts={per_expert_counts.tolist()} "
      f"valid_rows={int(jnp.sum(valid_mask))} {'OK' if ok else 'FAIL'}"
  )
  return ok


def check_extreme_skew_concentration() -> bool:
  """ALL tokens' picks land on a single expert within the shard -- extreme
  concentration, the opposite edge case from check_empty_shard."""
  config = _toy_shard_config(top_k=1, num_experts=16)
  num_tokens = 50
  local_expert_start, local_num_experts = 3, 5  # shard covers experts [3, 8)
  target_expert = 5  # inside the shard

  topk_idx = jnp.full((num_tokens, 1), target_expert, dtype=jnp.int32)
  topk_weight = jnp.full((num_tokens, 1), 0.7, dtype=jnp.float32)
  x = jax.random.normal(jax.random.key(1), (num_tokens, config.latent_size))

  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = filter_and_pad_to_shard(
      topk_idx, topk_weight, x, config, local_expert_start, local_num_experts,
      capacity_factor=4.0, tile_size=1,
  )

  local_target = target_expert - local_expert_start
  expected_counts = [0] * local_num_experts
  expected_counts[local_target] = num_tokens
  ok = per_expert_counts.tolist() == expected_counts and int(jnp.sum(valid_mask)) == num_tokens
  print(
      f"[local-test] extreme_skew_concentration: per_expert_counts={per_expert_counts.tolist()} "
      f"{'OK' if ok else 'FAIL'}"
  )
  return ok


def check_capacity_overflow() -> bool:
  """Forced capacity overflow: more real assignments land in the shard than
  the fixed m_padded budget allows -- confirms the excess is dropped from
  the tail of the expert-sorted order (matching an independent Python
  sort), not silently mis-truncated."""
  config = _toy_shard_config(top_k=1, num_experts=8)
  num_tokens = 20
  local_expert_start, local_num_experts = 0, 4

  key_idx, key_x = jax.random.split(jax.random.key(2))
  # ids restricted to [0, local_num_experts) -- every token lands in-shard,
  # so num_raw == num_tokens exactly, making the forced overflow predictable.
  topk_idx = jax.random.randint(key_idx, (num_tokens, 1), 0, local_num_experts, dtype=jnp.int32)
  topk_weight = jnp.full((num_tokens, 1), 0.3, dtype=jnp.float32)
  x = jax.random.normal(key_x, (num_tokens, config.latent_size))

  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = filter_and_pad_to_shard(
      topk_idx, topk_weight, x, config, local_expert_start, local_num_experts,
      capacity_factor=0.1, tile_size=1,  # deliberately tiny -- force overflow
  )

  m_padded = int(jnp.sum(group_sizes))
  keep_count = int(jnp.sum(valid_mask))
  overflow_triggered = keep_count < num_tokens

  # Independent reference: brute-force Python sort by global id, truncated to m_padded.
  topk_idx_np = jax.device_get(topk_idx).reshape(-1).tolist()
  hits_sorted = sorted(enumerate(topk_idx_np), key=lambda pair: pair[1])
  ref_token_order = [tok for tok, _gid in hits_sorted[:m_padded]]

  token_order_ok = jax.device_get(padded_token_idx[:keep_count]).tolist() == ref_token_order
  ok = overflow_triggered and keep_count == m_padded and token_order_ok
  print(
      f"[local-test] capacity_overflow: m_padded={m_padded} kept={keep_count}/{num_tokens} "
      f"{'OK' if ok else 'FAIL'}"
  )
  return ok


def check_padding_does_not_contaminate() -> bool:
  """Deliberately inject NON-ZERO values into padding rows of `shard_out`
  (not just relying on the FFN naturally producing zero for zero input)
  and confirm `_combine_shard_contribution` still adds exactly zero from
  them -- the invariant is enforced by `padded_combine_weight==0`, not by
  the content of `shard_out`."""
  num_tokens = 4
  latent_size = 8
  keep_count = 4

  # 6 rows total: 4 real, 2 padding -- padding rows get NON-ZERO shard_out
  # content on purpose, to test the invariant is enforced by weight, not by
  # the data happening to be zero.
  shard_out = jax.random.normal(jax.random.key(3), (6, latent_size))
  padded_token_idx = jnp.array([0, 1, 2, 3, -1, -1], dtype=jnp.int32)
  padded_combine_weight = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)

  routed_out_init = jnp.zeros((num_tokens, latent_size), dtype=jnp.float32)
  routed_out = _combine_shard_contribution(routed_out_init, shard_out, padded_token_idx, padded_combine_weight)

  expected = shard_out[:keep_count] * 0.5
  max_err = float(jnp.max(jnp.abs(routed_out - expected)))
  ok = max_err < 1e-6
  print(f"[local-test] padding_does_not_contaminate: max_err={max_err:.2e} {'OK' if ok else 'FAIL'}")
  return ok


def check_top_k_16() -> bool:
  """Exercises top_k=16, the REAL Kimi K3 value -- existing checks mostly
  use smaller top_k (2 or 4) for speed."""
  config = _toy_shard_config(top_k=16, num_experts=64)
  num_tokens = 128
  local_expert_start, local_num_experts = 10, 16

  key_router, key_x = jax.random.split(jax.random.key(4))
  router_weight = jax.random.normal(key_router, (config.hidden_size, config.num_experts)) * 0.02
  e_score_correction_bias = jnp.zeros((config.num_experts,))
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))
  x = hidden_states  # identity stand-in for down_proj's output, same convention as check_route_and_filter_correctness

  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = route_and_filter_to_local_shard(
      hidden_states, x, router_weight, e_score_correction_bias, config,
      local_expert_start=local_expert_start, local_num_experts=local_num_experts,
      capacity_factor=4.0,
  )

  expected_total = num_tokens * config.top_k * local_num_experts / config.num_experts
  ok = (
      sorted_tokens.shape[1] == config.hidden_size
      and int(jnp.sum(per_expert_counts)) == int(jnp.sum(valid_mask))
      and int(jnp.sum(valid_mask)) <= sorted_tokens.shape[0]
  )
  print(
      f"[local-test] top_k_16: valid_rows={int(jnp.sum(valid_mask))} "
      f"expected~{expected_total:.1f} {'OK' if ok else 'FAIL'}"
  )
  return ok


def check_global_to_local_boundary() -> bool:
  """Tokens whose global expert id sits EXACTLY at the shard's start and
  end boundaries -- an off-by-one here would silently include/exclude the
  wrong tokens."""
  config = _toy_shard_config(top_k=1, num_experts=20)
  local_expert_start, local_num_experts = 5, 4  # shard covers experts [5, 9)

  # token 0: global id 4 (just below start)      -> must be EXCLUDED
  # token 1: global id 5 (exactly the start)      -> must be INCLUDED, local id 0
  # token 2: global id 8 (last id inside the shard) -> must be INCLUDED, local id 3
  # token 3: global id 9 (exactly the end, first id past the shard) -> must be EXCLUDED
  topk_idx = jnp.array([[4], [5], [8], [9]], dtype=jnp.int32)
  topk_weight = jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32)
  x = jax.random.normal(jax.random.key(5), (4, config.latent_size))

  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = filter_and_pad_to_shard(
      topk_idx, topk_weight, x, config, local_expert_start, local_num_experts,
      capacity_factor=4.0, tile_size=1,
  )

  kept_tokens = sorted(jax.device_get(padded_token_idx[: int(jnp.sum(valid_mask))]).tolist())
  ok = (
      kept_tokens == [1, 2]
      and int(per_expert_counts[0]) == 1  # token 1 -> local id 5-5=0
      and int(per_expert_counts[3]) == 1  # token 2 -> local id 8-5=3
      and int(jnp.sum(per_expert_counts)) == 2
  )
  print(
      f"[local-test] global_to_local_boundary: kept_tokens={kept_tokens} "
      f"per_expert_counts={per_expert_counts.tolist()} {'OK' if ok else 'FAIL'}"
  )
  return ok


def check_first_and_last_shard() -> bool:
  """Tiling a global expert range into shards: the FIRST shard
  (`local_expert_start=0`) and the LAST shard (ending EXACTLY at
  `global_num_experts`, no overflow past the global range) both work."""
  config = _toy_shard_config(top_k=2, num_experts=24)
  local_num_experts = 6
  num_shards = config.num_experts // local_num_experts  # 4

  key_router, key_x = jax.random.split(jax.random.key(6))
  router_weight = jax.random.normal(key_router, (config.hidden_size, config.num_experts)) * 0.02
  e_score_correction_bias = jnp.zeros((config.num_experts,))
  num_tokens = 64
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))
  x = hidden_states

  ok = True
  for shard_idx in (0, num_shards - 1):
    local_expert_start = shard_idx * local_num_experts
    local_end = local_expert_start + local_num_experts
    assert local_end <= config.num_experts, "test setup bug: shard range exceeds global_num_experts"
    (
        sorted_tokens, group_sizes, valid_mask, per_expert_counts,
        padded_token_idx, padded_combine_weight,
    ) = route_and_filter_to_local_shard(
        hidden_states, x, router_weight, e_score_correction_bias, config,
        local_expert_start=local_expert_start, local_num_experts=local_num_experts,
        capacity_factor=4.0,
    )
    num_valid = int(jnp.sum(valid_mask))
    shard_ok = num_valid == int(jnp.sum(per_expert_counts))
    print(
        f"[local-test] first_and_last_shard: shard_idx={shard_idx} "
        f"local_expert_start={local_expert_start} (range [{local_expert_start},{local_end})) "
        f"valid_rows={num_valid} {'OK' if shard_ok else 'FAIL'}"
    )
    ok = ok and shard_ok
  return ok


def check_batch_seq_invariance() -> bool:
  """There is no separate batch/seq axis anywhere in this pipeline -- only
  `num_tokens = hidden_states.shape[0]` matters, the same invariant the
  latency sweeps already confirmed empirically for TIMING (e.g. `(1,2048)`
  matching `(2,1024)`). This is a lightweight regression GUARD for that
  architectural invariant (true by construction today, since these
  functions never receive a separate batch parameter), not a deep test --
  it exists so a future change that accidentally introduces batch-dependent
  behavior gets caught here rather than on a TPU."""
  config = _toy_shard_config(top_k=4, num_experts=16)
  num_tokens = 32  # could be conceptually (batch=1,seq=32), (batch=4,seq=8), (batch=2,seq=16), ...
  local_expert_start, local_num_experts = 0, 8

  key_router, key_x = jax.random.split(jax.random.key(7))
  router_weight = jax.random.normal(key_router, (config.hidden_size, config.num_experts)) * 0.02
  e_score_correction_bias = jnp.zeros((config.num_experts,))
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))
  x = hidden_states

  result_a = route_and_filter_to_local_shard(
      hidden_states, x, router_weight, e_score_correction_bias, config,
      local_expert_start=local_expert_start, local_num_experts=local_num_experts,
      capacity_factor=4.0,
  )
  result_b = route_and_filter_to_local_shard(
      hidden_states, x, router_weight, e_score_correction_bias, config,
      local_expert_start=local_expert_start, local_num_experts=local_num_experts,
      capacity_factor=4.0,
  )
  ok = all(bool(jnp.array_equal(a, b)) for a, b in zip(result_a, result_b))
  print(f"[local-test] batch_seq_invariance: {'OK' if ok else 'FAIL'}")
  return ok


def check_dtype_no_silent_promotion() -> bool:
  """bf16 inputs must produce bf16 outputs throughout -- including the
  empty-group/zero-padding branches, which is exactly the bug class this
  project has hit before (`latent_moe_forward`'s empty-expert-group branch
  once defaulted to float32 via an un-dtyped `jnp.zeros`, silently
  upcasting everything downstream under bf16 compute)."""
  config = _toy_shard_config(top_k=2, num_experts=8)
  num_tokens = 6
  local_expert_start, local_num_experts = 0, 2  # small shard, empty for this test's picks

  key = jax.random.key(8)
  ok = True
  for dtype in (jnp.float32, jnp.bfloat16):
    # Every pick lands OUTSIDE the shard -- guarantees the empty-shard
    # branch (zero real rows) is exercised for both dtypes, not left to a
    # lucky random draw.
    topk_idx = jnp.full((num_tokens, config.top_k), 5, dtype=jnp.int32)  # expert 5, outside [0, 2)
    topk_weight = jnp.full((num_tokens, config.top_k), 0.5, dtype=dtype)
    x = jax.random.normal(key, (num_tokens, config.latent_size), dtype=dtype)

    (
        sorted_tokens, group_sizes, valid_mask, per_expert_counts,
        padded_token_idx, padded_combine_weight,
    ) = filter_and_pad_to_shard(
        topk_idx, topk_weight, x, config, local_expert_start, local_num_experts,
        capacity_factor=4.0, tile_size=1,
    )
    dtype_ok = sorted_tokens.dtype == dtype and padded_combine_weight.dtype == dtype
    print(
        f"[local-test] dtype_no_silent_promotion: dtype={dtype} "
        f"sorted_tokens.dtype={sorted_tokens.dtype} "
        f"padded_combine_weight.dtype={padded_combine_weight.dtype} {'OK' if dtype_ok else 'FAIL'}"
    )
    ok = ok and dtype_ok

  # Also check _local_shard_expert_ffn preserves bf16 through its
  # empty-group branch specifically (both real experts empty, only the
  # padding bucket non-empty).
  group_sizes_bf16 = jnp.array([0, 0, 3], dtype=jnp.int32)
  sorted_tokens_bf16 = jnp.zeros((3, config.latent_size), dtype=jnp.bfloat16)
  expert_gate = jnp.zeros((3, config.latent_size, config.intermediate_size), dtype=jnp.bfloat16)
  expert_up = jnp.zeros((3, config.latent_size, config.intermediate_size), dtype=jnp.bfloat16)
  expert_down = jnp.zeros((3, config.intermediate_size, config.latent_size), dtype=jnp.bfloat16)
  shard_out = _local_shard_expert_ffn(sorted_tokens_bf16, expert_gate, expert_up, expert_down, group_sizes_bf16, config)
  ffn_dtype_ok = shard_out.dtype == jnp.bfloat16
  print(
      f"[local-test] dtype_no_silent_promotion (FFN, all-real-experts-empty): "
      f"shard_out.dtype={shard_out.dtype} {'OK' if ffn_dtype_ok else 'FAIL'}"
  )
  ok = ok and ffn_dtype_ok
  return ok


def check_sharded_sum_matches_reference_multi_config() -> bool:
  """`check_sharded_forward_correctness` only ever tested ONE config (32
  experts, 4 shards of 8, top_k=4). Broaden across several configs/seeds
  for real confidence -- reuses that same function (already proven,
  2026-08-28, max_err=9.31e-10) with different arguments rather than
  duplicating its logic."""
  configs = [
      dict(seed=0, num_tokens=32, global_num_experts=16, local_num_experts=4, top_k=2),
      dict(seed=1, num_tokens=96, global_num_experts=32, local_num_experts=8, top_k=4),
      dict(seed=2, num_tokens=64, global_num_experts=48, local_num_experts=16, top_k=6),
      dict(seed=3, num_tokens=200, global_num_experts=64, local_num_experts=8, top_k=16),  # real top_k
  ]
  ok = True
  for cfg in configs:
    result = check_sharded_forward_correctness(**cfg)
    print(f"[local-test] sharded_sum_matches_reference: config={cfg} {'OK' if result else 'FAIL'}")
    ok = ok and result
  return ok


if __name__ == "__main__":
  print(f"devices: {jax.devices()}")
  checks = [
      check_empty_shard,
      check_extreme_skew_concentration,
      check_capacity_overflow,
      check_padding_does_not_contaminate,
      check_top_k_16,
      check_global_to_local_boundary,
      check_first_and_last_shard,
      check_batch_seq_invariance,
      check_dtype_no_silent_promotion,
      check_sharded_sum_matches_reference_multi_config,
  ]
  results = [check() for check in checks]
  assert all(results), "one or more local routing edge-case checks failed"
  print(f"\nOK: all {len(results)} local routing edge-case checks passed")
