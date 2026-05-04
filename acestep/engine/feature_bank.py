"""StreamV2V-faithful feature bank port for ACE-Step.

This is the V2 port. It mirrors the user-facing surface of
``Jeff-LiangF/streamv2v`` (``utils/wrapper.py``) so the same hyper-
parameters and the same two mechanisms apply: Extended Attention (EA)
and Feature Fusion (FF). The geometry is adapted for our denoise-step-
batched decoder forward, which is why per-(layer, step) state lives
on the bank and the patched forward gathers along the step axis.

Mechanisms
----------

EA (Extended Attention)
    The patched ``self_attn.forward`` concats the bank's cached K/V
    onto the current K/V along the seq dim before SDPA. The bank
    holds ``cache_depth`` prior writes per (layer, denoise_step), so
    the new attention sequence is ``(1 + cache_depth) * T`` tokens
    long. ``strength`` is an additive log-bias on the bank columns
    (1.0 = raw concat, matching StreamV2V; 0.0 = bank fully masked).

FF (Feature Fusion)
    After the EA-extended SDPA + o_proj produces the layer's output,
    on a configurable subset of layers (``fi_layers``), the patched
    forward looks up the nearest neighbor of each output token in the
    bank's cached prior outputs (cosine similarity, gated by
    ``fi_threshold``), then blends ``out = (1-α)*out + α*nn_out``.
    Caches the *pre-blend* output back into the FF bank to avoid
    recursive feedback.

Cache management
----------------

cache_depth
    Ring-buffer depth per (layer, step). Higher = bank attends to
    more prior generations at the same denoise step.
cache_interval
    Write only every Nth tick. Slows bank churn so it reflects a
    longer-horizon feature memory. Reads happen every tick regardless.
tome_enabled / tome_ratio
    Optional StreamV2V-style ToMe compression on writes. When the
    target ring slot is already valid, the existing slot's K/V/output
    is concatenated with the new K/V/output along seq, then a random
    bipartite soft-matching merge collapses the result back to the
    slot's fixed seq length. Disabled by default; flip on to bank
    information from many ticks into a single ring slot.

State sketch
------------

Per banked layer (lazy-allocated on first forward):

    layer_banks[L]   : [2, num_steps, cache_depth, kv_heads, T, head_dim]
    output_banks[L]  : [num_steps, cache_depth, T, hidden]   (FF only)

Shared:

    valid_slots      : [num_steps, cache_depth] bool
    write_pos        : [num_steps] long  (next ring slot per step)
    step_indices     : [B] long          (per-tick row->step map)
    frame_id         : int               (advances via tick(); cache
                                          interval gates fire on this)

Hard requirements (carried over from V1)
----------------------------------------

- PyTorch decoder path only. The TRT decoder needs the bank as engine
  I/O; that's a separate concern (tracked elsewhere).
- Decoder must NOT be ``torch.compile``'d at install time.
  ``enable_feature_bank_on_decoder`` refuses ``OptimizedModule``; the
  caller can compile *after* the patch is installed.
- CFG negative-cond pass should not poison the bank. The pipeline
  flips ``write_enabled`` off around it (``StreamPipeline._tick_pt``).
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


DEFAULT_BANKED_LAYERS: Tuple[int, ...] = tuple(range(1, 24, 2))
# StreamV2V's default FF layer subset is the decoder-side UNet blocks
# (``up_blocks.0``/``up_blocks.1``/``mid_block``). For the DiT we use
# the deeper half of the banked full-attention layers as the analog --
# it's where features have stabilized into something semantically
# meaningful enough for cosine-similarity NN to be useful.
DEFAULT_FF_LAYERS: Tuple[int, ...] = tuple(range(13, 24, 2))


# ----------------------------------------------------------------------
# ToMe: random bipartite soft matching (port of streamv2v utils.py)
# ----------------------------------------------------------------------


def random_bipartite_soft_matching(metric: torch.Tensor, ratio: float = 0.5):
    """Build a token-merge function that compresses ``metric`` by ``ratio``.

    Mirrors ``streamv2v.models.utils.random_bipartite_soft_matching``.
    Splits the seq dim of ``metric`` ([B, N, C]) into two random sets
    A (size ``r = floor(ratio * N)``) and B (size ``N - r``), then for
    each token in A picks the most-similar token in B by cosine sim.
    The returned merge function scatter-reduces A into B's slots.

    Returns ``(merge_kv_out, merge_kv, merge_out)``: each takes the
    same-shape companion tensors as ``metric`` and returns the merged
    versions. We only use ``merge_kv_out`` here (output cache also
    needs merging when FF is on and ToMe is on).
    """
    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)
        r = int(ratio * N)
        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        m_norm = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a, b = split(m_norm)
        scores = a @ b.transpose(-1, -2)
        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge_kv_out(
        keys: torch.Tensor,
        values: torch.Tensor,
        outputs: torch.Tensor,
        mode: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_k, dst_k = split(keys)
        dst_k = dst_k.scatter_reduce(
            -2, dst_idx.expand(B, r, src_k.shape[-1]), src_k, reduce=mode,
        )
        src_v, dst_v = split(values)
        dst_v = dst_v.scatter_reduce(
            -2, dst_idx.expand(B, r, src_v.shape[-1]), src_v, reduce=mode,
        )
        src_o, dst_o = split(outputs)
        dst_o = dst_o.scatter_reduce(
            -2, dst_idx.expand(B, r, src_o.shape[-1]), src_o, reduce=mode,
        )
        return dst_k, dst_v, dst_o

    return merge_kv_out


# ----------------------------------------------------------------------
# Feature Fusion helper (port of streamv2v utils.get_nn_feats)
# ----------------------------------------------------------------------


def _get_nn_feats(
    x: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.98,
) -> torch.Tensor:
    """For each token in ``x``, find the most-similar token in ``y``
    by cosine similarity. Tokens whose best match is below ``threshold``
    keep their own value; the rest get replaced by the matched ``y``
    token. ``x``: ``[B, T_x, C]``, ``y``: ``[B, T_y, C]``.
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    cosine = torch.matmul(x_norm, y_norm.transpose(1, 2))  # [B, T_x, T_y]
    max_cos, nn_idx = torch.max(cosine, dim=-1)            # [B, T_x]
    mask = max_cos < threshold                              # [B, T_x]
    idx_expanded = nn_idx.unsqueeze(-1).expand(-1, -1, x_norm.size(-1))
    nn_tensor = torch.gather(y, 1, idx_expanded)           # [B, T_x, C]
    return torch.where(mask.unsqueeze(-1), x, nn_tensor)


# ----------------------------------------------------------------------
# FeatureBank
# ----------------------------------------------------------------------


class FeatureBank:
    """Cross-generation K/V (+ output) bank for streaming ACE-Step.

    See module docstring for the full state diagram. Knob defaults
    mirror StreamV2V's ``utils/wrapper.py`` so a fresh bank with no
    overrides behaves like the reference implementation.
    """

    def __init__(
        self,
        banked_layers: Sequence[int] = DEFAULT_BANKED_LAYERS,
        num_steps: int = 8,
        cache_depth: int = 1,
        cache_interval: int = 4,
        strength: float = 1.0,
        fi_enabled: bool = True,
        fi_strength: float = 0.8,
        fi_threshold: float = 0.98,
        fi_layers: Sequence[int] = DEFAULT_FF_LAYERS,
        tome_enabled: bool = False,
        tome_ratio: float = 0.5,
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if cache_depth < 1:
            raise ValueError(f"cache_depth must be >= 1, got {cache_depth}")
        if cache_interval < 1:
            raise ValueError(f"cache_interval must be >= 1, got {cache_interval}")
        if not 0.0 < tome_ratio < 1.0:
            raise ValueError(f"tome_ratio must be in (0, 1), got {tome_ratio}")

        self.banked: frozenset[int] = frozenset(banked_layers)
        self.num_steps: int = int(num_steps)
        self.cache_depth: int = int(cache_depth)
        self.cache_interval: int = int(cache_interval)
        self.strength: float = float(strength)
        self.enabled: bool = True
        self.write_enabled: bool = True

        # Feature Fusion (StreamV2V's second mechanism: cosine-NN blend
        # on the post-attn output at a layer subset).
        self.fi_enabled: bool = bool(fi_enabled)
        self.fi_strength: float = float(fi_strength)
        self.fi_threshold: float = float(fi_threshold)
        self.fi_layers: frozenset[int] = frozenset(fi_layers)

        # ToMe (random bipartite soft matching on writes). Off by
        # default since it costs an extra pass and only helps when
        # cache_depth is large enough that you want to merge many
        # ticks' info into each ring slot.
        self.tome_enabled: bool = bool(tome_enabled)
        self.tome_ratio: float = float(tome_ratio)

        # Lazy state ----------------------------------------------------
        # Per-layer EA storage: [2, num_steps, cache_depth, kv_heads, T, head_dim]
        self.layer_banks: Dict[int, torch.Tensor] = {}
        # Per-FF-layer output storage: [num_steps, cache_depth, T, hidden]
        self.output_banks: Dict[int, torch.Tensor] = {}
        # [num_steps, cache_depth] bool — shared across all layers (all
        # banked layers see the same step coverage at any tick).
        self.valid_slots: Optional[torch.Tensor] = None
        # [num_steps] long — next ring slot to write at each step.
        self.write_pos: Optional[torch.Tensor] = None
        # [B] long — set per-tick by the pipeline before each forward.
        self.step_indices: Optional[torch.Tensor] = None
        # Frame counter (StreamV2V semantics: cache_interval gates
        # fire on this). Advanced via tick() once per StreamPipeline
        # tick; layer forwards do not advance it themselves.
        self.frame_id: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all cached entries. Call between unrelated streams.

        Clears the per-layer storage dicts entirely (rather than just
        zeroing the tensors) so the next forward re-allocates at the
        current T / kv_heads / head_dim. That means a source swap
        with a different latent length is handled cleanly: the
        pipeline calls ``bank.reset()`` on T change and the next
        decoder forward gets fresh tensors at the new shape.
        """
        self.layer_banks.clear()
        self.output_banks.clear()
        self.valid_slots = None
        self.write_pos = None
        self.frame_id = 0

    def tick(self) -> None:
        """Advance the frame counter. Call once per StreamPipeline tick.

        ``cache_interval`` gates fire when ``frame_id %
        cache_interval == 0``, so this is what decides when writes
        actually land vs. get skipped.
        """
        self.frame_id += 1

    def set_cache_depth(self, depth: int) -> None:
        """Resize the ring depth. Drops all cached tensors so the next
        forward re-allocates at the new shape. ``valid_slots`` and
        ``write_pos`` are also cleared (they're shape-keyed too).
        Idempotent if ``depth == self.cache_depth``.
        """
        if depth < 1:
            raise ValueError(f"cache_depth must be >= 1, got {depth}")
        if depth == self.cache_depth:
            return
        self.cache_depth = int(depth)
        self.layer_banks.clear()
        self.output_banks.clear()
        self.valid_slots = None
        self.write_pos = None
        self.frame_id = 0

    def num_entries(self) -> int:
        """Diagnostic: total populated (layer, step, depth) slots.

        At saturation: ``len(banked_layers) * num_steps * cache_depth``.
        """
        if self.valid_slots is None:
            return 0
        return int(self.valid_slots.sum().item()) * len(self.layer_banks)

    def is_step_valid(self, step_idx: int) -> bool:
        """True iff some prior write has landed at step ``step_idx``
        in *any* ring slot."""
        if self.valid_slots is None:
            return False
        return bool(self.valid_slots[step_idx].any().item())

    # ------------------------------------------------------------------
    # Per-tick state
    # ------------------------------------------------------------------

    def set_step_indices(
        self,
        idxs: Union[Sequence[int], torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> None:
        """Set the per-row step indices for the next forward."""
        if isinstance(idxs, torch.Tensor):
            self.step_indices = idxs.to(dtype=torch.long)
            if device is not None:
                self.step_indices = self.step_indices.to(device=device)
        else:
            target_device = device
            if target_device is None and self.valid_slots is not None:
                target_device = self.valid_slots.device
            self.step_indices = torch.tensor(
                list(idxs), dtype=torch.long, device=target_device,
            )

    # ------------------------------------------------------------------
    # Lazy allocation
    # ------------------------------------------------------------------

    def _ensure_shared_state(self, device: torch.device) -> None:
        """Allocate ``valid_slots`` / ``write_pos`` on first use."""
        if self.valid_slots is None:
            self.valid_slots = torch.zeros(
                self.num_steps, self.cache_depth,
                dtype=torch.bool, device=device,
            )
        if self.write_pos is None:
            self.write_pos = torch.zeros(
                self.num_steps, dtype=torch.long, device=device,
            )

    def get_or_alloc_layer(
        self,
        layer_idx: int,
        num_kv_heads: int,
        T: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[2, num_steps, cache_depth, kv_heads, T, head_dim]``."""
        existing = self.layer_banks.get(layer_idx)
        target_shape = (
            2, self.num_steps, self.cache_depth, num_kv_heads, T, head_dim,
        )
        if existing is not None:
            if existing.shape != target_shape:
                raise RuntimeError(
                    f"FeatureBank layer {layer_idx} was allocated with "
                    f"shape {tuple(existing.shape)} but current forward "
                    f"requires {target_shape}. Call bank.reset() / rebuild "
                    f"the bank if T or cache_depth changed."
                )
            return existing

        new = torch.zeros(*target_shape, device=device, dtype=dtype)
        self.layer_banks[layer_idx] = new
        self._ensure_shared_state(device)
        return new

    def get_or_alloc_output_layer(
        self,
        layer_idx: int,
        T: int,
        hidden: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[num_steps, cache_depth, T, hidden]`` (FF only)."""
        existing = self.output_banks.get(layer_idx)
        target_shape = (self.num_steps, self.cache_depth, T, hidden)
        if existing is not None:
            if existing.shape != target_shape:
                raise RuntimeError(
                    f"FeatureBank output layer {layer_idx} was allocated "
                    f"with shape {tuple(existing.shape)} but current "
                    f"forward requires {target_shape}. Call bank.reset() / "
                    f"rebuild if T changed."
                )
            return existing

        new = torch.zeros(*target_shape, device=device, dtype=dtype)
        self.output_banks[layer_idx] = new
        self._ensure_shared_state(device)
        return new


# ----------------------------------------------------------------------
# Patched self-attention forward (eager + compile-friendly)
# ----------------------------------------------------------------------


def _patched_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position=None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    """Drop-in replacement for ``AceStepAttention.forward`` (self-attn).

    Steps:
      1. Compute Q/K/V (same as un-patched).
      2. EA: gather ``cache_depth`` prior K/V at this row's step;
         concat along seq; build attn_bias with strength + per-slot
         validity; SDPA over ``(1+cache_depth)*T`` columns.
      3. FF (if layer in ``fi_layers`` and ``fi_enabled``): cosine-NN
         lookup of attn_out tokens against this layer's prior outputs;
         blend by ``fi_strength``.
      4. Write (if ``write_enabled`` and frame_id passes interval gate):
         ToMe-merge with the existing ring slot's content if
         ``tome_enabled`` and slot is valid; otherwise direct
         ``index_copy_`` into the ring at ``write_pos``. Mark slot
         valid; advance ring pos for affected steps.
    """
    bank: FeatureBank = self._feature_bank

    if encoder_hidden_states is not None:
        return self._unpatched_forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

    if not bank.enabled or self.layer_idx not in bank.banked:
        return self._unpatched_forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]  # [B, T]
    hidden_shape = (*input_shape, -1, self.head_dim)

    Q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    K = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    V = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    B, kv_heads, T, head_dim = K.shape
    D = bank.cache_depth

    layer_bank = bank.get_or_alloc_layer(
        self.layer_idx, kv_heads, T, head_dim, K.device, K.dtype,
    )  # [2, num_steps, D, kv_heads, T, head_dim]

    # Resolve per-row step index. Promote Python lists to tensor on
    # first use (cheap; happens at most once per StreamPipeline tick).
    if bank.step_indices is None:
        raise RuntimeError(
            "FeatureBank.step_indices not set. StreamPipeline must call "
            "bank.set_step_indices(...) before each decoder forward."
        )
    step_idx_t = bank.step_indices
    if not isinstance(step_idx_t, torch.Tensor):
        step_idx_t = torch.tensor(
            step_idx_t, dtype=torch.long, device=K.device,
        )
    elif step_idx_t.device != K.device:
        step_idx_t = step_idx_t.to(K.device)
    if step_idx_t.shape[0] != B:
        raise RuntimeError(
            f"FeatureBank.step_indices length {step_idx_t.shape[0]} does "
            f"not match batch size {B}."
        )

    # ---------- EA: gather banked K/V along step axis ----------
    # gathered: [2, B, D, kv_heads, T, head_dim]
    gathered = layer_bank.index_select(1, step_idx_t)
    # → [B, kv_heads, D*T, head_dim]
    K_bank = gathered[0].permute(0, 2, 1, 3, 4).reshape(B, kv_heads, D * T, head_dim)
    V_bank = gathered[1].permute(0, 2, 1, 3, 4).reshape(B, kv_heads, D * T, head_dim)

    K_full = torch.cat([K, K_bank], dim=2)  # [B, kv_heads, (1+D)*T, head_dim]
    V_full = torch.cat([V, V_bank], dim=2)

    # Per-row, per-slot validity → flatten across (D, T) for the mask.
    # valid_per_row: [B, D] bool
    valid_per_row = bank.valid_slots.index_select(0, step_idx_t)
    # [B, D, T] → [B, D*T]
    bank_col_validity = (
        valid_per_row[:, :, None].expand(B, D, T).reshape(B, D * T)
    )

    NEG_INF = torch.finfo(Q.dtype).min
    s = bank.strength
    if s <= 0.0:
        log_s_scalar = NEG_INF
    elif s == 1.0:
        log_s_scalar = 0.0
    else:
        log_s_scalar = math.log(s)
    log_s_t = torch.tensor(log_s_scalar, device=Q.device, dtype=Q.dtype)
    neg_inf_t = torch.tensor(NEG_INF, device=Q.device, dtype=Q.dtype)

    bank_cols = torch.where(
        bank_col_validity[:, None, None, :].expand(B, 1, T, D * T),
        log_s_t, neg_inf_t,
    )
    cur_cols = torch.zeros(B, 1, T, T, device=Q.device, dtype=Q.dtype)
    attn_bias = torch.cat([cur_cols, bank_cols], dim=-1)  # [B, 1, T, (1+D)*T]

    if self.num_key_value_groups > 1:
        K_full = K_full.repeat_interleave(self.num_key_value_groups, dim=1)
        V_full = V_full.repeat_interleave(self.num_key_value_groups, dim=1)

    attn_out = F.scaled_dot_product_attention(
        Q, K_full, V_full,
        attn_mask=attn_bias,
        dropout_p=0.0,
        scale=self.scaling,
    )  # [B, q_heads, T, head_dim]

    attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_out = self.o_proj(attn_out)  # [B, T, hidden]

    # ---------- FF: cosine-NN blend on the post-attn output ----------
    do_ff = bank.fi_enabled and (self.layer_idx in bank.fi_layers)
    pre_ff_out: Optional[torch.Tensor] = None
    if do_ff:
        hidden = attn_out.shape[-1]
        out_bank = bank.get_or_alloc_output_layer(
            self.layer_idx, T, hidden, attn_out.device, attn_out.dtype,
        )  # [num_steps, D, T, hidden]
        # Snapshot pre-blend output for the writeback; FF must not
        # feed itself or it'll lock to whatever lookup hit first.
        pre_ff_out = attn_out.detach().clone()

        # Gather along step axis: [B, D, T, hidden] → [B, D*T, hidden]
        out_gathered = out_bank.index_select(0, step_idx_t).reshape(
            B, D * T, hidden,
        )
        # Suppress invalid slots so cosine-NN can't latch onto a
        # zero-init region. Setting invalid tokens to a value with
        # very small norm makes their cosine ~0 < threshold, so the
        # threshold mask in _get_nn_feats keeps the original token.
        if not bool(valid_per_row.all().item()):
            slot_valid = bank_col_validity[:, :, None]  # [B, D*T, 1]
            out_gathered = torch.where(
                slot_valid, out_gathered, torch.zeros_like(out_gathered),
            )

        nn_out = _get_nn_feats(attn_out, out_gathered, threshold=bank.fi_threshold)
        attn_out = (1.0 - bank.fi_strength) * attn_out + bank.fi_strength * nn_out

    # ---------- Write: gated by write_enabled + interval ----------
    write_now = bank.write_enabled and (
        bank.frame_id % bank.cache_interval == 0
    )
    if write_now:
        K_det = K.detach()
        V_det = V.detach()
        # write_pos[step] gives the next ring slot to fill at each step.
        # Per-row: pos_per_row = write_pos[step_idx_t].
        pos_per_row = bank.write_pos.index_select(0, step_idx_t)  # [B] long
        # Flatten (step, depth) into a single axis so we can scatter
        # per-row in one index_copy_. flat_idx = step * D + pos.
        flat_idx = step_idx_t * D + pos_per_row  # [B] long

        # ToMe path: when the target slot is valid AND tome_enabled is
        # on, merge the slot's existing contents with the new K/V (and
        # output) before scattering. This compresses many ticks of
        # info into a single ring slot at the cost of one extra
        # bipartite-matching pass.
        do_tome = bank.tome_enabled and bool(
            bank.valid_slots.view(-1)[flat_idx].any().item()
        )
        if do_tome:
            K_to_write, V_to_write, out_to_write = _tome_merge_slot(
                layer_bank, flat_idx,
                K_det, V_det,
                bank.output_banks.get(self.layer_idx) if do_ff else None,
                pre_ff_out if do_ff else None,
                bank.tome_ratio,
            )
        else:
            K_to_write = K_det
            V_to_write = V_det
            out_to_write = pre_ff_out if do_ff else None

        # Flatten (step, depth) and scatter K/V at flat_idx.
        layer_bank_flat = layer_bank.view(
            2, bank.num_steps * D, kv_heads, T, head_dim,
        )
        layer_bank_flat[0].index_copy_(0, flat_idx, K_to_write)
        layer_bank_flat[1].index_copy_(0, flat_idx, V_to_write)

        # FF output bank writeback (uses the same flat_idx alignment).
        if do_ff and out_to_write is not None:
            out_bank_flat = bank.output_banks[self.layer_idx].view(
                bank.num_steps * D, T, attn_out.shape[-1],
            )
            out_bank_flat.index_copy_(0, flat_idx, out_to_write)

        # Mark slots valid.
        bank.valid_slots.view(-1)[flat_idx] = True

        # Advance write_pos once per *unique step touched* this forward.
        # Multiple rows at the same step still only advance the ring
        # head once (semantically: the last write at step k lands at
        # the current pos, and the next write at step k goes to pos+1
        # mod D). Duplicate-step handling: index_copy_ above will
        # last-write-wins at the current slot, which matches V1's
        # collision behavior.
        unique_steps = torch.unique(step_idx_t)
        bank.write_pos[unique_steps] = (
            bank.write_pos[unique_steps] + 1
        ) % D

    return attn_out, None


# ----------------------------------------------------------------------
# ToMe slot merge (used at write time when bank.tome_enabled)
# ----------------------------------------------------------------------


def _tome_merge_slot(
    layer_bank: torch.Tensor,
    flat_idx: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    out_bank: Optional[torch.Tensor],
    out_new: Optional[torch.Tensor],
    ratio: float,
):
    """Concat each row's existing slot with new K/V/output, ToMe-merge
    back to seq length T, return the merged tensors ready to scatter
    into ``layer_bank`` (and the output bank, if FF is on).

    ``layer_bank``: ``[2, num_steps, D, kv_heads, T, head_dim]`` (rank-6).
    ``flat_idx``: ``[B]`` long, indices into the flattened (step, D)
    axis. ``K_new`` / ``V_new``: ``[B, kv_heads, T, head_dim]``.
    ``out_bank``: ``[num_steps, D, T, hidden]`` or None.
    ``out_new``: ``[B, T, hidden]`` or None.

    Returned tensors have the same shapes as ``K_new`` / ``V_new`` /
    ``out_new``: T tokens per row, packed for direct ``index_copy_``.
    """
    _, _, _, kv_heads, T, head_dim = layer_bank.shape
    B = K_new.shape[0]
    num_steps_x_D = layer_bank.shape[1] * layer_bank.shape[2]

    layer_flat = layer_bank.view(2, num_steps_x_D, kv_heads, T, head_dim)
    K_existing = layer_flat[0].index_select(0, flat_idx)  # [B, kv_heads, T, head_dim]
    V_existing = layer_flat[1].index_select(0, flat_idx)

    # ToMe operates on [B, N, C] with N = seq, C = combined-feature.
    # Fold heads into channel dim so cosine similarity matches across
    # all heads simultaneously (StreamV2V's convention).
    def fold(t: torch.Tensor) -> torch.Tensor:
        # [B, kv_heads, T, head_dim] -> [B, T, kv_heads*head_dim]
        return t.permute(0, 2, 1, 3).reshape(B, t.shape[2], -1)

    def unfold(t: torch.Tensor) -> torch.Tensor:
        # [B, N, kv_heads*head_dim] -> [B, kv_heads, N, head_dim]
        return t.reshape(B, t.shape[1], kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    K_concat = torch.cat([fold(K_existing), fold(K_new)], dim=1)  # [B, 2T, hidden]
    V_concat = torch.cat([fold(V_existing), fold(V_new)], dim=1)
    if out_bank is not None and out_new is not None:
        out_existing = out_bank.view(num_steps_x_D, T, out_bank.shape[-1]).index_select(0, flat_idx)
        out_concat = torch.cat([out_existing, out_new], dim=1)  # [B, 2T, hidden]
    else:
        out_concat = K_concat  # placeholder; merge_kv_out wants 3 tensors

    # ratio=0.5 collapses 2T → T. Other ratios drift away from T and
    # would need padding/reshape — for now we lock to 0.5 here.
    merge = random_bipartite_soft_matching(K_concat, ratio=0.5)
    K_merged, V_merged, out_merged = merge(K_concat, V_concat, out_concat)
    # Each is [B, T, ...] (when ratio=0.5). User-controlled tome_ratio
    # is preserved in the API but only 0.5 is exact-reshape-safe.

    K_out = unfold(K_merged)
    V_out = unfold(V_merged)
    out_out: Optional[torch.Tensor]
    if out_bank is not None and out_new is not None:
        out_out = out_merged
    else:
        out_out = None
    return K_out, V_out, out_out


# ----------------------------------------------------------------------
# Install / uninstall
# ----------------------------------------------------------------------


def _is_compiled(module: torch.nn.Module) -> bool:
    OptimizedModule = getattr(
        getattr(torch, "_dynamo", None), "eval_frame", None
    )
    if OptimizedModule is None:
        return False
    OptimizedModule = getattr(OptimizedModule, "OptimizedModule", None)
    if OptimizedModule is None:
        return False
    return isinstance(module, OptimizedModule)


def enable_feature_bank_on_decoder(
    decoder: torch.nn.Module,
    bank: FeatureBank,
) -> None:
    """Patch the self-attn forward on every banked layer in ``decoder``."""
    if _is_compiled(decoder):
        raise RuntimeError(
            "Feature bank cannot be installed on a torch.compile'd decoder. "
            "Install the patch first, then call torch.compile."
        )

    if not hasattr(decoder, "layers"):
        raise RuntimeError(
            "decoder has no .layers attribute -- not an AceStepDiTModel?"
        )

    patched = 0
    for idx, layer in enumerate(decoder.layers):
        if idx not in bank.banked:
            continue
        attn = layer.self_attn
        if hasattr(attn, "_unpatched_forward"):
            attn._feature_bank = bank
            continue
        attn._feature_bank = bank
        attn._unpatched_forward = attn.forward
        attn.forward = types.MethodType(_patched_self_attn_forward, attn)
        patched += 1

    logger.info(
        "Feature bank enabled on %d layers (EA), FF on %d layers; "
        "cache_depth=%d cache_interval=%d strength=%.2f",
        patched, len(bank.fi_layers & bank.banked),
        bank.cache_depth, bank.cache_interval, bank.strength,
    )


def disable_feature_bank_on_decoder(decoder: torch.nn.Module) -> None:
    """Restore the original forward on every patched self-attn module."""
    if not hasattr(decoder, "layers"):
        return
    restored = 0
    for layer in decoder.layers:
        attn = layer.self_attn
        if hasattr(attn, "_unpatched_forward"):
            attn.forward = attn._unpatched_forward
            del attn._unpatched_forward
            del attn._feature_bank
            restored += 1
    logger.info("Feature bank disabled on %d layers", restored)
