# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# KV Cache Shared Flow-Matching Action Head
# Implements MiRobot-style attention sharing with MindVLM

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from vlaflow.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim].
            Then, if q and k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    """Simple RoPE implementation compatible with VLM RoPE"""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids):
        """
        Args:
            x: (B, seq_len, head_dim)
            position_ids: (B, seq_len)
        Returns:
            (cos, sin) each of shape (B, 1, seq_len, head_dim//2)
        """
        batch_size, seq_len, head_dim = x.shape
        device = x.device
        dtype = x.dtype

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        return cos, sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    """Attention that can attend to VLM KV cache + DiT self attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int = None,
        head_dim: int = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if head_dim is None:
            head_dim = hidden_size // num_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_heads

        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_past_key_values: tuple,
        position_embeddings: tuple,
        attention_mask: torch.Tensor = None,
    ):
        batch_size, q_len, _ = hidden_states.shape

        # ── 1. Project ──────────────────────────────────────────────
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # ── 2. Reshape to multi-head ────────────────────────────────
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads,    q_len, head_dim]

        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_kv_heads, q_len, head_dim]

        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_kv_heads, q_len, head_dim]

        # ── 3. Per-head norm ────────────────────────────────────────
        query_states = self.q_norm(query_states)  # [B, num_heads,    q_len, head_dim]
        key_states = self.k_norm(key_states)  # [B, num_kv_heads, q_len, head_dim]

        # ── 4. RoPE ─────────────────────────────────────────────────
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ── 5. 处理 VLM KV cache ────────────────────────────────────
        vlm_k_cache, vlm_v_cache = vlm_past_key_values
        # vlm_k_cache: [B, vlm_num_kv_heads, vlm_len, head_dim]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # 现在 key_states: [B, num_heads, q_len, head_dim]

        vlm_k_cache = repeat_kv(vlm_k_cache, self.num_key_value_groups)
        vlm_v_cache = repeat_kv(vlm_v_cache, self.num_key_value_groups)
        # 现在 vlm_k_cache: [B, num_heads, vlm_len, head_dim]

        # ── 6. 拼接 VLM cache + DiT KV ──────────────────────────────
        key_states = torch.cat([vlm_k_cache, key_states], dim=2)
        value_states = torch.cat([vlm_v_cache, value_states], dim=2)
        # key_states: [B, num_heads, vlm_len + dit_len, head_dim]

        # ── 7. Attention ─────────────────────────────────────────────
        attn_output = F.scaled_dot_product_attention(
            query=query_states,  # [B, num_heads, q_len,             head_dim]
            key=key_states,  # [B, num_heads, vlm_len + q_len,   head_dim]
            value=value_states,  # [B, num_heads, vlm_len + q_len,   head_dim]
            attn_mask=attention_mask,
            dropout_p=0.0,
        )

        # ── 8. Output projection ─────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        return self.o_proj(attn_output)


class DiTLayer(nn.Module):
    """DiT decoder layer with AdaLN and KV cache attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int = None,
        head_dim: int = None,
        intermediate_size: int = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.hidden_size = hidden_size
        self.attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # AdaLN parameters
        # self.adaln_gate_attn = nn.Parameter(torch.zeros(hidden_size))
        # self.adaln_gate_mlp = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_past_key_values: tuple,
        position_embeddings: tuple,
        t_embeddings: torch.Tensor,  # [B, hidden_size]
        attention_mask: torch.Tensor = None,
    ):
        # exit(0)
        # AdaLN: modulate with time embeddings
        # gate_attn = self.adaln_gate_attn + t_embeddings
        # gate_mlp = self.adaln_gate_mlp + t_embeddings
        (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp) = t_embeddings.chunk(6, dim=-1)

        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)

        attn_output = self.attn(
            hidden_states=hidden_states,
            vlm_past_key_values=vlm_past_key_values,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + F.silu(gate_attn).unsqueeze(1) * attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + F.silu(gate_mlp).unsqueeze(1) * mlp_output

        return hidden_states


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=False),  # 6 for 6 modulations
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        t: [B] continuous timesteps
        Returns: [B, 6 * hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # [B, 6 * hidden_size]


class KVSharedDiT(nn.Module):
    """DiT with KV cache sharing with VLM (MiRobot style)"""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads: int = None,
        head_dim: int = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [
                DiTLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.t_embedder = TimestepEmbedding(hidden_size)

        self.rotary_emb = RotaryPositionalEmbedding(
            dim=head_dim if head_dim else hidden_size // num_heads,
            max_position_embeddings=2048,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, dit_len, hidden_size]
        vlm_past_key_values: list,  # List of tuples (k_cache, v_cache) per layer
        t_continuous: torch.Tensor,  # [B] continuous timesteps
        dit_position_ids: torch.Tensor,  # [B, dit_len] DiT position IDs
        attention_mask: torch.Tensor = None,  # [B, 1, dit_len, vlm_len+dit_len]
    ):
        # Time embedding
        t_embeddings = self.t_embedder(t_continuous)  # [B, 6 * hidden_size]

        # RoPE embeddings for DiT positions
        position_embeddings = self.rotary_emb(hidden_states, dit_position_ids)

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                vlm_past_key_values=vlm_past_key_values[layer_idx],
                position_embeddings=position_embeddings,
                t_embeddings=t_embeddings,
                attention_mask=attention_mask,
            )

        return hidden_states


@dataclass
class KVSharedFlowmatchingActionHeadConfig(PretrainedConfig):
    """Configuration for KV shared flow-matching action head"""

    hidden_size: int = field(default=2048, metadata={"help": "Hidden size of VLM"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension"})
    state_dim: int = field(default=7, metadata={"help": "State dimension"})
    action_horizon: int = field(default=8, metadata={"help": "Action horizon"})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": ""})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep buckets"})
    num_inference_timesteps: int = field(default=4, metadata={"help": "Number of inference steps"})
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens"})
    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})

    # DiT configuration
    dit_num_layers: int = field(default=16, metadata={"help": "Number of DiT layers (<= VLM layers)"})
    dit_num_heads: int = field(default=32, metadata={"help": "Number of attention heads"})
    dit_num_key_value_heads: int = field(default=None, metadata={"help": "Number of KV heads for GQA"})
    dit_head_dim: int = field(default=64, metadata={"help": "Head dimension"})
    dit_rope_theta: float = field(default=10000.0, metadata={"help": "RoPE theta"})
    vlm_kv_layer_offset: int = field(
        default=None, metadata={"help": "Offset from end of VLM layers. If None, uses last dit_num_layers VLM layers."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class KVSharedFlowmatchingActionHead(nn.Module):
    """
    Flow-matching action head with KV cache sharing between VLM and DiT.

    Key differences from LayerwiseFlowmatchingActionHead:
    - Uses KV cache sharing (like MiRobot) instead of cross-attention
    - DiT can attend to VLM's KV cache directly
    - More efficient attention mechanism
    """

    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model

        # VLM hidden size (used only for KV sharing — action expert can differ)
        self.hidden_size = global_config.framework.qwenvl.vl_hidden_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        # Determine action expert internal hidden dim.
        # Priority: explicit action_hidden_dim > hidden_dim_factor * vl_hidden_dim > vl_hidden_dim (default 1.0)
        _factor = float(getattr(action_config, "hidden_dim_factor", 1.0))
        _explicit = getattr(action_config, "action_hidden_dim", None)
        if _explicit is not None:
            self.action_hidden_dim = int(_explicit)
        else:
            self.action_hidden_dim = int(self.hidden_size * _factor)

        # DiT with KV sharing — uses action_hidden_dim internally; head_dim must match VLM KV head_dim
        self.dit = KVSharedDiT(
            hidden_size=self.action_hidden_dim,
            num_layers=action_config.dit_num_layers,
            num_heads=action_config.dit_num_heads,
            num_key_value_heads=action_config.dit_num_key_value_heads,
            head_dim=action_config.dit_head_dim,
            rope_theta=action_config.dit_rope_theta,
        )

        # State encoder
        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.action_hidden_dim,
            )
            if action_config.state_dim
            else None
        )

        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.action_hidden_dim,
        )

        # Action decoder
        self.action_decoder = MLP(
            input_dim=self.action_hidden_dim,
            hidden_dim=max(self.action_hidden_dim, 1024),
            output_dim=self.action_dim,
        )

        # Sink token (like MiRobot)
        # self.sink_token = nn.Parameter(torch.randn(1, 1, self.action_hidden_dim) * 0.02)
        self.action_config = action_config

        if action_config.num_target_vision_tokens > 0:
            # Future tokens
            self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.action_hidden_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # Position embedding
        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(2048, self.action_hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Flow matching noise distribution
        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

        # VLM KV layer selection
        self.dit_num_layers = action_config.dit_num_layers
        self.vlm_kv_layer_offset = action_config.vlm_kv_layer_offset

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_attention_mask(
        self,
        vlm_attention_mask: torch.Tensor,  # [B, vlm_input_len] 原始输入mask
        dit_len: int,
        device: torch.device,
        vlm_kv_len: int = None,
        prefix_len: int = 0,  # state + future_tokens 的 token 数；其余均为 action tokens
    ) -> torch.Tensor:
        """
        Build attention mask for DiT with KV cache sharing.
        Shape: [B, 1, dit_len, vlm_kv_len + dit_len]

        Visibility rules:
          - All DiT tokens → VLM KV:        full (all visible)
          - prefix tokens  → prefix tokens: causal (lower-triangular)
          - prefix tokens  → action tokens: blocked
          - action tokens  → DiT tokens:    full bidirectional (all visible)
        """
        batch_size = vlm_attention_mask.shape[0]

        if vlm_kv_len is None:
            vlm_kv_len = vlm_attention_mask.shape[1]

        action_len = dit_len - prefix_len

        # ── VLM cache 部分：DiT 所有 token 均可见 ──────────────────
        vlm_mask = torch.ones(batch_size, dit_len, vlm_kv_len, device=device, dtype=torch.bool)

        # ── DiT 自注意力部分 ────────────────────────────────────────
        if prefix_len > 0:
            # prefix 行（state + future tokens）：因果下三角
            prefix_causal = torch.tril(torch.ones(batch_size, prefix_len, prefix_len, device=device, dtype=torch.bool))
            # prefix 行 → action 列：不可见
            prefix_to_action = torch.zeros(batch_size, prefix_len, action_len, device=device, dtype=torch.bool)
            top = torch.cat([prefix_causal, prefix_to_action], dim=-1)  # [B, prefix_len, dit_len]

        # action 行：对全部 DiT token 双向可见
        action_to_all = torch.ones(batch_size, action_len, dit_len, device=device, dtype=torch.bool)

        if prefix_len > 0:
            dit_mask = torch.cat([top, action_to_all], dim=1)  # [B, dit_len, dit_len]
        else:
            dit_mask = action_to_all  # [B, action_len, action_len]

        # 拼接: [B, dit_len, vlm_kv_len + dit_len]
        full_mask = torch.cat([vlm_mask, dit_mask], dim=-1)
        full_mask = full_mask.unsqueeze(1)  # [B, 1, dit_len, vlm_kv_len + dit_len]

        # True → 0.0 (可见), False → -inf (遮蔽)
        float_mask = torch.zeros_like(full_mask, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~full_mask, float("-inf"))

        return float_mask  # [B, 1, dit_len, vlm_kv_len + dit_len]

    def _prepare_dit_input(
        self,
        noisy_action: torch.Tensor,  # [B, T, action_dim]
        state: torch.Tensor = None,  # [B, 1, state_dim] or None
    ) -> torch.Tensor:
        """Prepare DiT input: sink + state + future_tokens + action"""
        batch_size = noisy_action.shape[0]

        # Sink token
        # sink = self.sink_token.weight.expand(batch_size, -1, -1)

        # State embedding
        state_embed = self.state_encoder(state) if state is not None else None

        future_tokens = None
        if self.action_config.num_target_vision_tokens > 0:
            # Future tokens
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate
        if state_embed is not None and future_tokens is not None:
            hidden_states = torch.cat([state_embed, future_tokens, noisy_action], dim=1)
        elif state_embed is not None and future_tokens is None:
            hidden_states = torch.cat([state_embed, noisy_action], dim=1)
        elif future_tokens is not None and state_embed is None:
            hidden_states = torch.cat([future_tokens, noisy_action], dim=1)
        else:
            hidden_states = noisy_action

        return hidden_states  # [B, dit_len, hidden_size]

    def forward(
        self,
        vlm_past_key_values: list,  # List of (k_cache, v_cache) per layer
        vlm_attention_mask: torch.Tensor,  # [B, vlm_len]
        actions: torch.Tensor,  # [B, T, action_dim]
        state: torch.Tensor = None,  # [B, 1, state_dim]
        action_mask: torch.Tensor = None,  # [B, T, action_dim], 1=real, 0=padded
    ) -> torch.Tensor:
        """
        Forward pass with KV cache sharing.

        Args:
            vlm_past_key_values: KV cache from VLM
            vlm_attention_mask: Attention mask from VLM
            actions: Target actions
            state: Robot state
            action_mask: Optional per-element validity mask (1.0 where the
                action target is real, 0.0 where the dataloader zero-padded
                or inserted a sentinel — typical for single-arm datasets
                routed to the shared 14-D dual-arm space). When provided,
                the flow-matching MSE is averaged only over the valid slots.

        Returns:
            loss: Flow matching loss
        """
        device = actions.device
        batch_size = actions.shape[0]

        # Sample timestep
        t_continuous = self.sample_time(batch_size, device, actions.dtype)  # [B]
        t_discrete = (t_continuous * self.num_timestep_buckets).long()

        # Add noise (flow matching)
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        noisy_trajectory = (1 - t_continuous[:, None, None]) * noise + t_continuous[:, None, None] * actions
        velocity = actions - noise

        # Encode action
        action_features = self.action_encoder(noisy_trajectory, t_discrete)

        # Add positional embedding
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Prepare DiT input
        hidden_states = self._prepare_dit_input(action_features, state)  # [B, dit_len, hidden_size]

        # Build attention mask
        dit_len = hidden_states.shape[1]
        # attention_mask = self._build_attention_mask(vlm_attention_mask, dit_len, device)

        # DiT position IDs (continue from VLM positions)
        vlm_max_pos = vlm_attention_mask.sum(dim=1).max().item()
        dit_position_ids = (
            torch.arange(dit_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) + vlm_max_pos + 1
        )

        # Select VLM KV cache for DiT layers
        # If vlm_kv_layer_offset is None, use last dit_num_layers VLM layers
        # Otherwise, use last (vlm_kv_layer_offset + dit_num_layers) VLM layers
        vlm_num_layers = len(vlm_past_key_values)
        if self.vlm_kv_layer_offset is not None:
            vlm_kv_start_idx = vlm_num_layers - self.vlm_kv_layer_offset - self.dit_num_layers
            if vlm_kv_start_idx < 0:
                raise ValueError(
                    f"vlm_kv_layer_offset ({self.vlm_kv_layer_offset}) + dit_num_layers ({self.dit_num_layers}) "
                    f"exceeds VLM layers ({vlm_num_layers})"
                )
        else:
            vlm_kv_start_idx = vlm_num_layers - self.dit_num_layers
            if vlm_kv_start_idx < 0:
                raise ValueError(f"dit_num_layers ({self.dit_num_layers}) cannot exceed VLM layers ({vlm_num_layers})")

        # exit()
        # Extract the relevant VLM KV cache for DiT layers
        def extract_kv_from_cache(vlm_past_key_values, start_idx, num_layers):
            """
            从各种格式的 KV cache 中提取指定层的 (k, v) 元组列表

            Returns:
                List of (k_tensor, v_tensor) tuples, length = num_layers
            """
            selected_kv = []

            for i in range(start_idx, start_idx + num_layers):
                layer_cache = vlm_past_key_values[i]

                # 判断类型，提取 k/v 张量
                if isinstance(layer_cache, tuple):
                    # 普通 tuple: (k, v)
                    k, v = layer_cache

                elif hasattr(layer_cache, "key_cache") and hasattr(layer_cache, "value_cache"):
                    # 某些版本的 DynamicLayer
                    k = layer_cache.key_cache
                    v = layer_cache.value_cache

                elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                    # 另一种属性命名
                    k = layer_cache.keys
                    v = layer_cache.values

                elif hasattr(layer_cache, "key") and hasattr(layer_cache, "value"):
                    k = layer_cache.key
                    v = layer_cache.value

                else:
                    raise TypeError(
                        f"Unknown cache layer type: {type(layer_cache)}\n"
                        f"Available attributes: {[a for a in dir(layer_cache) if not a.startswith('_')]}"
                    )

                selected_kv.append((k, v))

            return selected_kv

        selected_vlm_kv = extract_kv_from_cache(
            vlm_past_key_values, start_idx=vlm_kv_start_idx, num_layers=self.dit_num_layers
        )

        # selected_vlm_kv[0] = (k, v), k.shape = [B, num_kv_heads, vlm_kv_len, head_dim]
        vlm_kv_len = selected_vlm_kv[0][0].shape[2]

        prefix_len = (1 if state is not None else 0) + self.action_config.num_target_vision_tokens
        attention_mask = self._build_attention_mask(
            vlm_attention_mask,
            dit_len,
            device,
            vlm_kv_len=vlm_kv_len,
            prefix_len=prefix_len,
        )

        # Run DiT with KV cache sharing
        hidden_states = self.dit(
            hidden_states=hidden_states,
            vlm_past_key_values=selected_vlm_kv,
            t_continuous=t_continuous,
            dit_position_ids=dit_position_ids,
            attention_mask=attention_mask,
        )

        # Decode actions (take only action portion)
        pred_actions = self.action_decoder(hidden_states[:, -actions.shape[1] :])

        # Flow matching loss
        sq_err = (pred_actions - velocity) ** 2
        if action_mask is not None:
            # Same dtype as sq_err to avoid dtype-promotion in autocast contexts.
            mask_f = action_mask.to(sq_err.dtype)
            denom = mask_f.sum().clamp_min(1.0)
            loss = (sq_err * mask_f).sum() / denom
        else:
            loss = sq_err.mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vlm_past_key_values: list,
        vlm_attention_mask: torch.Tensor,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference with diffusion sampling"""
        batch_size = vlm_attention_mask.shape[0]
        device = vlm_attention_mask.device

        # Initialize with random noise
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Pre-compute DiT position IDs and attention mask
        dit_len = (1 if state is not None else 0) + self.action_horizon + self.action_config.num_target_vision_tokens
        # exit(0)
        vlm_max_pos = vlm_attention_mask.sum(dim=1).max().item()
        dit_position_ids = (
            torch.arange(dit_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) + vlm_max_pos + 1
        )
        # attention_mask = self._build_attention_mask(vlm_attention_mask, dit_len, device)

        # Select VLM KV cache for DiT layers (same as in forward)
        vlm_num_layers = len(vlm_past_key_values)
        if self.vlm_kv_layer_offset is not None:
            vlm_kv_start_idx = vlm_num_layers - self.vlm_kv_layer_offset - self.dit_num_layers
        else:
            vlm_kv_start_idx = vlm_num_layers - self.dit_num_layers
        # selected_vlm_kv = vlm_past_key_values[vlm_kv_start_idx:]

        def extract_kv_from_cache(vlm_past_key_values, start_idx, num_layers):
            """
            从各种格式的 KV cache 中提取指定层的 (k, v) 元组列表

            Returns:
                List of (k_tensor, v_tensor) tuples, length = num_layers
            """
            selected_kv = []

            for i in range(start_idx, start_idx + num_layers):
                layer_cache = vlm_past_key_values[i]  # 整数索引，合法

                # 判断类型，提取 k/v 张量
                if isinstance(layer_cache, tuple):
                    # 普通 tuple: (k, v)
                    k, v = layer_cache

                elif hasattr(layer_cache, "key_cache") and hasattr(layer_cache, "value_cache"):
                    # 某些版本的 DynamicLayer
                    k = layer_cache.key_cache
                    v = layer_cache.value_cache

                elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                    # 另一种属性命名
                    k = layer_cache.keys
                    v = layer_cache.values

                elif hasattr(layer_cache, "key") and hasattr(layer_cache, "value"):
                    k = layer_cache.key
                    v = layer_cache.value

                else:
                    raise TypeError(
                        f"Unknown cache layer type: {type(layer_cache)}\n"
                        f"Available attributes: {[a for a in dir(layer_cache) if not a.startswith('_')]}"
                    )

                selected_kv.append((k, v))

            return selected_kv

        selected_vlm_kv = extract_kv_from_cache(
            vlm_past_key_values, start_idx=vlm_kv_start_idx, num_layers=self.dit_num_layers
        )

        # selected_vlm_kv[0] = (k, v), k.shape = [B, num_kv_heads, vlm_kv_len, head_dim]
        vlm_kv_len = selected_vlm_kv[0][0].shape[2]

        prefix_len = (1 if state is not None else 0) + self.action_config.num_target_vision_tokens
        attention_mask = self._build_attention_mask(
            vlm_attention_mask,
            dit_len,
            device,
            vlm_kv_len=vlm_kv_len,
            prefix_len=prefix_len,
        )

        # Denoising loop
        for step in range(num_steps):
            t_continuous = torch.full(
                size=(batch_size,), fill_value=step / num_steps, device=device, dtype=torch.float32
            )
            t_discrete = (t_continuous * self.num_timestep_buckets).long()

            # Encode action
            action_features = self.action_encoder(actions, t_discrete)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Prepare DiT input
            hidden_states = self._prepare_dit_input(action_features, state)

            # Run DiT with selected VLM KV cache
            hidden_states = self.dit(
                hidden_states=hidden_states,
                vlm_past_key_values=selected_vlm_kv,  # Use selected KV cache
                t_continuous=t_continuous,
                dit_position_ids=dit_position_ids,
                attention_mask=attention_mask,
            )

            # Decode and integrate
            pred_velocity = self.action_decoder(hidden_states[:, -self.action_horizon :])
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_kv_shared_action_model(config=None):
    """
    Factory: build KV shared flow-matching action head from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        KVSharedFlowmatchingActionHead: Initialized action head.
    """
    return KVSharedFlowmatchingActionHead(global_config=config)
