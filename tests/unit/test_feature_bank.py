#!/usr/bin/env python3
"""Unit tests for the StreamV2V-style feature bank patch.

These tests run on CPU and don't load checkpoint weights -- they
construct a tiny ``AceStepAttention`` from a small synthetic config,
patch it via ``feature_bank``, and verify the read/write contract:

1. ``enable`` then forward with empty bank should produce output
   numerically equal to the un-patched forward (banked half is fully
   masked out, contributing zero to softmax).
2. After the first forward, the bank holds entries keyed by
   ``(layer_idx, step_idx)`` for every row in the batch.
3. A second forward with a populated bank should produce output that
   differs measurably from the un-patched forward (banked K/V now
   contributes to the attention sum).
4. ``write_enabled = False`` should leave the bank unchanged across
   a forward.
5. ``disable`` should restore the original forward behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="module")
def turbo_modules():
    """Return the vendored AceStepAttention / AceStepConfig + RoPE module."""
    from acestep.models.modeling_acestep_v15_turbo import AceStepAttention
    from acestep.models.configuration_acestep_v15 import AceStepConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    return AceStepAttention, AceStepConfig, Qwen3RotaryEmbedding


@pytest.fixture
def tiny_config(turbo_modules):
    _, AceStepConfig, _ = turbo_modules
    cfg = AceStepConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=1000000,
        use_sliding_window=True,
        sliding_window=64,
        # Mirror the v15-turbo default pattern: full_attention on odd
        # indices, sliding_attention on even.
        layer_types=[
            "full_attention" if (i % 2 == 1) else "sliding_attention"
            for i in range(4)
        ],
    )
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.fixture
def attn_pair(turbo_modules, tiny_config):
    """Build a single full_attention layer's self_attn and a RoPE module."""
    AceStepAttention, _, Qwen3RotaryEmbedding = turbo_modules
    torch.manual_seed(0)
    attn = AceStepAttention(
        tiny_config, layer_idx=1, is_cross_attention=False,
    ).eval()
    rope = Qwen3RotaryEmbedding(tiny_config)
    return attn, rope


def _make_inputs(tiny_config, B=3, T=16, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, T, tiny_config.hidden_size)
    pos_ids = torch.arange(T).unsqueeze(0)
    return x, pos_ids


def _patch(attn, bank):
    """Apply the feature-bank patch to a single AceStepAttention module."""
    import types
    from acestep.engine.feature_bank import _patched_self_attn_forward

    attn._feature_bank = bank
    attn._unpatched_forward = attn.forward
    attn.forward = types.MethodType(_patched_self_attn_forward, attn)


def _unpatch(attn):
    if hasattr(attn, "_unpatched_forward"):
        attn.forward = attn._unpatched_forward
        del attn._unpatched_forward
        del attn._feature_bank


@torch.no_grad()
def test_empty_bank_matches_unpatched(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    y_patched_empty, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    # With no bank entries, the banked half is fully masked. Output
    # must match the un-patched forward to within SDPA numeric noise.
    torch.testing.assert_close(
        y_patched_empty, y_unpatched, atol=1e-5, rtol=1e-4,
    )

    _unpatch(attn)


@torch.no_grad()
def test_first_forward_populates_bank(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 3, 5]
    _patch(attn, bank)

    assert bank.num_entries() == 0
    attn(x, attention_mask=None, position_embeddings=pos_emb)

    # Bank stores per (layer, step); writeback scatters batch rows into
    # the step axis at step_indices = [0, 3, 5].
    assert 1 in bank.layer_banks
    assert bank.is_step_valid(0)
    assert bank.is_step_valid(3)
    assert bank.is_step_valid(5)
    assert not bank.is_step_valid(1)
    assert bank.num_entries() == 3

    # Rank-6: [2, num_steps, cache_depth, kv_heads, T, head_dim].
    # bank[0, step, depth_slot] gives the K at that ring slot.
    layer_bank = bank.layer_banks[1]
    assert layer_bank.shape == (
        2, bank.num_steps, bank.cache_depth,
        tiny_config.num_key_value_heads, x.shape[1], tiny_config.head_dim,
    )
    K0, V0 = layer_bank[0, 0, 0], layer_bank[1, 0, 0]
    assert K0.shape == (tiny_config.num_key_value_heads, x.shape[1], tiny_config.head_dim)
    assert V0.shape == K0.shape

    _unpatch(attn)


@torch.no_grad()
def test_bank_with_identical_input_is_a_noop(attn_pair, tiny_config):
    """When banked K/V equals current K/V, the math degenerates.

    Doubling identical K positions splits softmax weight evenly across
    each pair (w/2 + w/2 = w), and identical V at both positions sums
    back to the same output as the un-patched forward. So the right
    way to verify the bank is *active* is the cross-song scenario
    (different inputs across writes/reads); see
    ``test_cross_song_handoff_roundtrip``. This test pins the
    degenerate-input invariant so we notice if it ever drifts.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    attn(x, attention_mask=None, position_embeddings=pos_emb)
    y_with_bank, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )
    torch.testing.assert_close(y_with_bank, y_unpatched, atol=1e-5, rtol=1e-4)

    _unpatch(attn)


@torch.no_grad()
def test_write_disabled_leaves_bank_unchanged(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    # Seed the bank with one forward.
    attn(x, attention_mask=None, position_embeddings=pos_emb)
    # Rank-6 storage: [k_or_v=0, step=0, depth=0]
    K0_before = bank.layer_banks[1][0, 0, 0].clone()

    # Run again with writes disabled.
    bank.write_enabled = False
    x2 = x + 0.5
    attn(x2, attention_mask=None, position_embeddings=pos_emb)

    K0_after = bank.layer_banks[1][0, 0, 0]
    torch.testing.assert_close(K0_after, K0_before)

    _unpatch(attn)


@torch.no_grad()
def test_disable_restores_original_forward(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_baseline, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    _unpatch(attn)
    assert not hasattr(attn, "_feature_bank")
    assert not hasattr(attn, "_unpatched_forward")

    y_after_unpatch, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )
    torch.testing.assert_close(y_after_unpatch, y_baseline)


@torch.no_grad()
def test_skips_non_banked_layers(turbo_modules, tiny_config):
    """Layer not in bank.banked must fall through to the original forward."""
    AceStepAttention, _, Qwen3RotaryEmbedding = turbo_modules
    from acestep.engine.feature_bank import FeatureBank

    torch.manual_seed(0)
    # layer_idx=0 is sliding_attention; we mark only layer 1 as banked.
    attn = AceStepAttention(
        tiny_config, layer_idx=0, is_cross_attention=False,
    ).eval()
    rope = Qwen3RotaryEmbedding(tiny_config)
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    y_patched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    # Layer 0 isn't in bank.banked -- patched forward must fall
    # through to the original. Output should match exactly (same
    # code path).
    torch.testing.assert_close(y_patched, y_unpatched)
    # And the bank must remain empty.
    assert bank.num_entries() == 0

    _unpatch(attn)


@torch.no_grad()
def test_step_indices_length_mismatch_raises(attn_pair, tiny_config):
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config, B=3)
    pos_emb = rope(x, pos_ids)

    bank = FeatureBank(banked_layers=(1,))
    bank.step_indices = [0, 1]  # length 2, but B=3 -- mismatch.
    _patch(attn, bank)

    with pytest.raises(RuntimeError, match="step_indices length"):
        attn(x, attention_mask=None, position_embeddings=pos_emb)

    _unpatch(attn)


@torch.no_grad()
def test_cross_song_handoff_roundtrip(attn_pair, tiny_config):
    """Simulate the StreamPipeline handoff: song A writes, song B reads.

    Pretend we have two consecutive 'songs' going through the
    pipeline. Song A occupies one batch row at step 3 on tick T;
    song B occupies a different batch row at step 3 on tick T+1.
    Song B's forward should see song A's K/V via the bank entry at
    ``(layer_idx=1, step=3)``.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    bank = FeatureBank(banked_layers=(1,))
    _patch(attn, bank)

    # Tick T: song A only, single row at step 3.
    torch.manual_seed(101)
    x_A = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb_A = rope(x_A, pos_ids)
    bank.step_indices = [3]
    attn(x_A, attention_mask=None, position_embeddings=pos_emb_A)
    assert 1 in bank.layer_banks
    assert bank.is_step_valid(3)
    KA = bank.layer_banks[1][0, 3, 0].clone()

    # Tick T+1: song B (different inputs), also at step 3, in row 0.
    torch.manual_seed(202)
    x_B = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb_B = rope(x_B, pos_ids)
    bank.step_indices = [3]

    # Run with bank temporarily disabled to capture B's "no bank"
    # baseline.
    bank.enabled = False
    y_B_no_bank, _ = attn(
        x_B, attention_mask=None, position_embeddings=pos_emb_B,
    )
    bank.enabled = True

    # The 'no bank' run still wrote nothing because the patched
    # forward fell through to the un-patched path. KA should be
    # untouched.
    torch.testing.assert_close(bank.layer_banks[1][0, 3, 0], KA)

    # Real banked run: song B reads KA, then overwrites with KB.
    y_B_with_bank, _ = attn(
        x_B, attention_mask=None, position_embeddings=pos_emb_B,
    )
    diff = (y_B_with_bank - y_B_no_bank).abs().max().item()
    assert diff > 1e-3, (
        f"Cross-song handoff produced no detectable change "
        f"(max-abs diff = {diff:.3e})."
    )

    # Bank now holds KB, not KA.
    KB = bank.layer_banks[1][0, 3, 0]
    assert (KB - KA).abs().max().item() > 1e-3

    _unpatch(attn)


# ----------------------------------------------------------------------
# V2 (StreamV2V-faithful) coverage: cache_depth, cache_interval, FF, ToMe
# ----------------------------------------------------------------------


@torch.no_grad()
def test_cache_depth_ring_rotates(attn_pair, tiny_config):
    """With cache_depth=3, three consecutive writes at the same step
    fill ring slots 0, 1, 2 in order. ``write_pos`` advances mod D.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    bank = FeatureBank(banked_layers=(1,), cache_depth=3, cache_interval=1)
    bank.step_indices = [3]
    _patch(attn, bank)

    # Write three different inputs at step 3.
    K_writes = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        x = torch.randn(1, 16, tiny_config.hidden_size)
        pos_emb = rope(x, pos_ids)
        attn(x, attention_mask=None, position_embeddings=pos_emb)
        # write_pos[3] points to the *next* slot, so the just-written
        # slot is (write_pos[3] - 1) mod D.
        last_slot = (bank.write_pos[3].item() - 1) % bank.cache_depth
        K_writes.append(bank.layer_banks[1][0, 3, last_slot].clone())

    # write_pos[3] should now be back to 0 after 3 writes (3 % 3 = 0).
    assert bank.write_pos[3].item() == 0
    # All three ring slots should be populated and distinct.
    assert bool(bank.valid_slots[3].all().item())
    KA, KB, KC = K_writes
    assert (KA - KB).abs().max().item() > 1e-3
    assert (KB - KC).abs().max().item() > 1e-3

    _unpatch(attn)


@torch.no_grad()
def test_cache_interval_gates_writes(attn_pair, tiny_config):
    """With cache_interval=4, only ticks where frame_id%4==0 write.

    Reads still happen every tick. ``tick()`` is what advances
    ``frame_id``; the patched forward does not advance it.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    bank = FeatureBank(banked_layers=(1,), cache_depth=1, cache_interval=4)
    bank.step_indices = [0]
    _patch(attn, bank)

    # frame_id=0: 0%4==0 → writes.
    torch.manual_seed(101)
    x0 = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb = rope(x0, pos_ids)
    attn(x0, attention_mask=None, position_embeddings=pos_emb)
    K0 = bank.layer_banks[1][0, 0, 0].clone()
    assert bool(bank.valid_slots[0, 0].item())

    # frame_id=1,2,3: writes blocked.
    for _ in range(3):
        bank.tick()
        torch.manual_seed(202)
        x = torch.randn(1, 16, tiny_config.hidden_size)
        pos_emb = rope(x, pos_ids)
        attn(x, attention_mask=None, position_embeddings=pos_emb)
        torch.testing.assert_close(bank.layer_banks[1][0, 0, 0], K0)

    # frame_id=4: writes again.
    bank.tick()
    torch.manual_seed(303)
    x4 = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb = rope(x4, pos_ids)
    attn(x4, attention_mask=None, position_embeddings=pos_emb)
    K4 = bank.layer_banks[1][0, 0, 0]
    assert (K4 - K0).abs().max().item() > 1e-3

    _unpatch(attn)


@torch.no_grad()
def test_cache_depth_extends_attended_seq(attn_pair, tiny_config):
    """At cache_depth=2, the attended seq length must be (1+2)*T;
    the bank must contribute distinct K/V from both ring slots.
    Sanity check: with cache_depth=2 and two prior writes at the
    same step, output differs from the unpatched forward and from
    the cache_depth=1 case.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    # Baseline: unpatched.
    torch.manual_seed(7)
    x_query = torch.randn(1, 16, tiny_config.hidden_size)
    pos_emb = rope(x_query, pos_ids)
    y_unpatched, _ = attn(
        x_query, attention_mask=None, position_embeddings=pos_emb,
    )

    # Seed bank with two distinct prior writes at step 0 with depth=2.
    bank = FeatureBank(banked_layers=(1,), cache_depth=2, cache_interval=1)
    bank.step_indices = [0]
    _patch(attn, bank)

    for seed in (11, 22):
        torch.manual_seed(seed)
        x_seed = torch.randn(1, 16, tiny_config.hidden_size)
        pos_emb_seed = rope(x_seed, pos_ids)
        attn(x_seed, attention_mask=None, position_embeddings=pos_emb_seed)

    assert bool(bank.valid_slots[0].all().item())  # both ring slots filled

    # Now query: forward with the original input. Should differ from
    # the unpatched baseline because the bank now contributes 2T extra
    # banked tokens to attention.
    y_with_depth2, _ = attn(
        x_query, attention_mask=None, position_embeddings=pos_emb,
    )
    diff = (y_with_depth2 - y_unpatched).abs().max().item()
    assert diff > 1e-3, (
        f"cache_depth=2 with populated bank produced no detectable "
        f"divergence from unpatched (max-abs={diff:.3e})."
    )

    _unpatch(attn)


@torch.no_grad()
def test_strength_zero_masks_bank(attn_pair, tiny_config):
    """``strength=0`` zeros out the bank's softmax mass. Even with a
    populated bank, output must match the unpatched forward.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    x, pos_ids = _make_inputs(tiny_config)
    pos_emb = rope(x, pos_ids)

    y_unpatched, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )

    bank = FeatureBank(banked_layers=(1,), cache_depth=2, cache_interval=1)
    bank.step_indices = [0, 1, 2]
    _patch(attn, bank)

    # Populate.
    torch.manual_seed(999)
    x_seed = torch.randn(*x.shape)
    pos_emb_seed = rope(x_seed, pos_ids)
    attn(x_seed, attention_mask=None, position_embeddings=pos_emb_seed)

    # strength=0 -> bank columns should be -inf in attn_bias -> softmax
    # masks them out entirely -> result == unpatched.
    bank.strength = 0.0
    y_str0, _ = attn(
        x, attention_mask=None, position_embeddings=pos_emb,
    )
    torch.testing.assert_close(y_str0, y_unpatched, atol=1e-5, rtol=1e-4)

    _unpatch(attn)


@torch.no_grad()
def test_ff_blend_changes_output(turbo_modules, tiny_config):
    """Feature Fusion: when the layer is in ``fi_layers``, the
    post-attn output gets blended with the nearest-neighbor lookup
    against the previous tick's cached output. Use *different* seed
    and query inputs so the cached output isn't trivially equal to
    the current output (which would make the blend a no-op).
    """
    AceStepAttention, _, Qwen3RotaryEmbedding = turbo_modules
    from acestep.engine.feature_bank import FeatureBank

    torch.manual_seed(0)
    attn = AceStepAttention(
        tiny_config, layer_idx=1, is_cross_attention=False,
    ).eval()
    rope = Qwen3RotaryEmbedding(tiny_config)

    # Two distinct inputs: ``x_seed`` populates the bank on tick T,
    # ``x_query`` is the input on tick T+1.
    torch.manual_seed(42)
    x_seed = torch.randn(1, 16, tiny_config.hidden_size)
    torch.manual_seed(99)
    x_query = torch.randn(1, 16, tiny_config.hidden_size)
    pos_ids = torch.arange(16).unsqueeze(0)
    pos_emb_seed = rope(x_seed, pos_ids)
    pos_emb_query = rope(x_query, pos_ids)

    # FF-off control: only EA. Same seed -> same query -> baseline.
    bank_off = FeatureBank(
        banked_layers=(1,), fi_enabled=False, cache_interval=1,
    )
    bank_off.step_indices = [0]
    _patch(attn, bank_off)
    attn(x_seed, attention_mask=None, position_embeddings=pos_emb_seed)
    y_ff_off, _ = attn(
        x_query, attention_mask=None, position_embeddings=pos_emb_query,
    )
    _unpatch(attn)

    # FF on: layer 1 in fi_layers, threshold=0 so the blend never
    # falls back to the original token (every position picks an NN).
    bank_on = FeatureBank(
        banked_layers=(1,), fi_enabled=True, fi_layers=(1,),
        fi_strength=0.9, fi_threshold=0.0, cache_interval=1,
    )
    bank_on.step_indices = [0]
    _patch(attn, bank_on)
    attn(x_seed, attention_mask=None, position_embeddings=pos_emb_seed)
    y_ff_on, _ = attn(
        x_query, attention_mask=None, position_embeddings=pos_emb_query,
    )
    _unpatch(attn)

    diff = (y_ff_on - y_ff_off).abs().max().item()
    assert diff > 1e-4, (
        f"FF blend produced no detectable change "
        f"(max-abs={diff:.3e})."
    )


@torch.no_grad()
def test_tome_preserves_slot_shape(attn_pair, tiny_config):
    """ToMe-on-write must collapse (existing+new) back to T tokens
    so the rank-6 storage shape is preserved. We can't easily check
    that the merge math is *correct*, but we can at least pin that
    the slot stays the same shape and stays valid after a ToMe write.
    """
    from acestep.engine.feature_bank import FeatureBank

    attn, rope = attn_pair
    pos_ids = torch.arange(16).unsqueeze(0)

    bank = FeatureBank(
        banked_layers=(1,),
        cache_depth=1,
        cache_interval=1,
        tome_enabled=True,
        tome_ratio=0.5,
    )
    bank.step_indices = [0]
    _patch(attn, bank)

    target_shape = (
        2, bank.num_steps, bank.cache_depth,
        tiny_config.num_key_value_heads, 16, tiny_config.head_dim,
    )

    # Two writes at the same step. First: direct (slot empty, ToMe
    # path skipped). Second: triggers ToMe-merge with existing slot.
    for seed in (4242, 7777):
        torch.manual_seed(seed)
        x = torch.randn(1, 16, tiny_config.hidden_size)
        pos_emb = rope(x, pos_ids)
        attn(x, attention_mask=None, position_embeddings=pos_emb)
        assert bank.layer_banks[1].shape == target_shape
        assert bool(bank.valid_slots[0, 0].item())

    _unpatch(attn)
