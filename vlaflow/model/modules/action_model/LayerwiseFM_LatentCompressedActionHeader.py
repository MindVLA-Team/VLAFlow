# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# KV Cache Shared Flow-Matching Action Head with Compressed Latent World Prediction
#
# Uses KV sharing with AvgPool-k4 on present and future V-JEPA 2 tokens.
# Pooling is stateless; future-latent loss is measured on the pooled targets.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig

from vlaflow.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

# ─────────────────────────────────────────────────────────────────────────────
# Verbatim helpers (copied from LayerwiseFM_LatentPredictActionHeader.py)
# ─────────────────────────────────────────────────────────────────────────────


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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


class LatentEncoder(nn.Module):
    """Encode present latents for predicting future latents (single-pass, no time conditioning)."""

    def __init__(self, latent_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, latents):
        return self.projector(latents)


class LatentDecoder(nn.Module):
    """Decode DiT hidden states back to latent space for future-frame prediction."""

    def __init__(self, latent_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, latent_dim),
        )

    def forward(self, hidden_latents):
        return self.projector(hidden_latents)


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
    inv_freq: torch.Tensor

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids):
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
        prefix_kv: tuple = None,
    ):
        batch_size, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        vlm_k_cache, vlm_v_cache = vlm_past_key_values
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        vlm_k_cache = repeat_kv(vlm_k_cache, self.num_key_value_groups)
        vlm_v_cache = repeat_kv(vlm_v_cache, self.num_key_value_groups)
        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            key_states = torch.cat([vlm_k_cache, prefix_k, key_states], dim=2)
            value_states = torch.cat([vlm_v_cache, prefix_v, value_states], dim=2)
        else:
            key_states = torch.cat([vlm_k_cache, key_states], dim=2)
            value_states = torch.cat([vlm_v_cache, value_states], dim=2)
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)
        return self.o_proj(attn_output)

    def get_kv(self, hidden_states: torch.Tensor, position_embeddings: tuple) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        key_states = (key_states * cos.unsqueeze(1)) + (rotate_half(key_states) * sin.unsqueeze(1))
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states


class DiTLayer(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_past_key_values: tuple,
        position_embeddings: tuple,
        t_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
        prefix_kv: tuple = None,
    ):
        (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp) = t_embeddings.chunk(6, dim=-1)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        attn_output = self.attn(
            hidden_states=hidden_states,
            vlm_past_key_values=vlm_past_key_values,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            prefix_kv=prefix_kv,
        )
        hidden_states = residual + F.silu(gate_attn).unsqueeze(1) * attn_output
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
            nn.Linear(hidden_size, 6 * hidden_size, bias=False),
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class KVSharedDiT(nn.Module):
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
        hidden_states: torch.Tensor,
        vlm_past_key_values: list,
        t_continuous: torch.Tensor,
        dit_position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        prefix_kv_cache: list = None,
    ):
        t_embeddings = self.t_embedder(t_continuous)
        position_embeddings = self.rotary_emb(hidden_states, dit_position_ids)
        for layer_idx, layer in enumerate(self.layers):
            layer_prefix_kv = prefix_kv_cache[layer_idx] if prefix_kv_cache is not None else None
            hidden_states = layer(
                hidden_states=hidden_states,
                vlm_past_key_values=vlm_past_key_values[layer_idx],
                position_embeddings=position_embeddings,
                t_embeddings=t_embeddings,
                attention_mask=attention_mask,
                prefix_kv=layer_prefix_kv,
            )
        return hidden_states

    def prefill_prefix(
        self,
        prefix_hidden_states: torch.Tensor,
        vlm_past_key_values: list,
        dit_position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> list:
        batch_size = prefix_hidden_states.shape[0]
        device = prefix_hidden_states.device
        t_continuous = torch.zeros(batch_size, device=device, dtype=prefix_hidden_states.dtype)
        t_embeddings = self.t_embedder(t_continuous)
        position_embeddings = self.rotary_emb(prefix_hidden_states, dit_position_ids)
        prefix_kv_cache = []
        hidden_states = prefix_hidden_states
        for layer_idx, layer in enumerate(self.layers):
            (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp) = t_embeddings.chunk(6, dim=-1)
            residual = hidden_states
            h_normed = layer.input_layernorm(hidden_states)
            h_normed = h_normed * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
            kv = layer.attn.get_kv(h_normed, position_embeddings)
            prefix_kv_cache.append(kv)
            hidden_states = layer(
                hidden_states=hidden_states,
                vlm_past_key_values=vlm_past_key_values[layer_idx],
                position_embeddings=position_embeddings,
                t_embeddings=t_embeddings,
                attention_mask=attention_mask,
                prefix_kv=None,
            )
        return prefix_kv_cache


@dataclass
class KVSharedFlowmatchingActionHeadConfig(PretrainedConfig):
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
    dit_num_layers: int = field(default=16, metadata={"help": "Number of DiT layers (<= VLM layers)"})
    dit_num_heads: int = field(default=32, metadata={"help": "Number of attention heads"})
    dit_num_key_value_heads: int = field(default=None, metadata={"help": "Number of KV heads for GQA"})
    dit_head_dim: int = field(default=64, metadata={"help": "Head dimension"})
    dit_rope_theta: float = field(default=10000.0, metadata={"help": "RoPE theta"})
    vlm_kv_layer_offset: int = field(default=None, metadata={"help": "Offset from end of VLM layers."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# ─────────────────────────────────────────────────────────────────────────────
# New classes: LatentCompressor and LatentExpander
# ─────────────────────────────────────────────────────────────────────────────


class LatentCompressor(nn.Module):
    """Stateless AvgPool-k4 used by the released MindLWPI model."""

    def __init__(self, latent_dim: int, compress_ratio: int, method: str):
        super().__init__()
        if method != "avgpool" or compress_ratio != 4:
            raise ValueError("The released MindLWPI model requires avgpool compression with ratio 4.")
        self.compress_ratio = compress_ratio

    def forward(self, latents):
        batch, tokens, dim = latents.shape
        if tokens % self.compress_ratio:
            raise ValueError("Latent token count must be divisible by four.")
        return latents.reshape(batch, tokens // self.compress_ratio, self.compress_ratio, dim).mean(dim=2)


class KVSharedLatentCompressedFlowmatchingActionHead(nn.Module):
    """
    Flow-matching action head with KV cache sharing + compressed future-frame latent
    prediction.

    Architecture (single DiT forward pass):
      Input sequence: [state(1)] + [compressed_latent_tokens(N//k)] + [action_tokens(T)]

    Compression reduces the latent sequence from N to N//k tokens before the DiT,
    lowering attention cost while preserving the world-model pretraining signal.

    Pooling averages four adjacent tokens; the future target uses the same pooling.
    Configuration requires latent_compress_ratio=4 and latent_compress_method="avgpool".
    """

    def __init__(self, global_config, latent_dim, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        latent_config = global_config.framework.latent_extractor

        self.hidden_size = global_config.framework.qwenvl.vl_hidden_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.latent_dim = latent_dim
        self.latent_loss_weight = float(getattr(latent_config, "latent_loss_weight", 1.0))
        self.action_loss_weight = float(getattr(action_config, "action_loss_weight", 1.0))

        _factor = float(getattr(action_config, "hidden_dim_factor", 1.0))
        _explicit = getattr(action_config, "action_hidden_dim", None)
        if _explicit is not None:
            self.action_hidden_dim = int(_explicit)
        else:
            self.action_hidden_dim = int(self.hidden_size * _factor)

        self.dit = KVSharedDiT(
            hidden_size=self.action_hidden_dim,
            num_layers=action_config.dit_num_layers,
            num_heads=action_config.dit_num_heads,
            num_key_value_heads=action_config.dit_num_key_value_heads,
            head_dim=action_config.dit_head_dim,
            rope_theta=action_config.dit_rope_theta,
        )

        self.state_encoder = (
            MLP(input_dim=action_config.state_dim, output_dim=self.action_hidden_dim)
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.action_hidden_dim,
        )

        self.action_decoder = MLP(
            input_dim=self.action_hidden_dim,
            hidden_dim=max(self.action_hidden_dim, 1024),
            output_dim=self.action_dim,
        )

        self.latent_encoder = LatentEncoder(latent_dim=latent_dim, hidden_size=self.action_hidden_dim)
        self.latent_decoder = LatentDecoder(latent_dim=latent_dim, hidden_size=self.action_hidden_dim)

        self.action_config = action_config

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(2048, self.action_hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

        self.dit_num_layers = action_config.dit_num_layers
        self.vlm_kv_layer_offset = action_config.vlm_kv_layer_offset

        # ── Latent compression (new) ───────────────────────────────────────
        self.compress_ratio = int(getattr(action_config, "latent_compress_ratio", 4))
        self.compress_method = str(getattr(action_config, "latent_compress_method", "avgpool"))

        self.latent_compressor = LatentCompressor(latent_dim, self.compress_ratio, self.compress_method)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _build_attention_mask(
        self,
        vlm_attention_mask: torch.Tensor,
        dit_len: int,
        device: torch.device,
        vlm_kv_len: int = None,
        state_len: int = 0,
        latent_len: int = 0,
        action_len: int = 0,
    ) -> torch.Tensor:
        """Build attention mask with latent isolation (same logic as original)."""
        batch_size = vlm_attention_mask.shape[0]
        if vlm_kv_len is None:
            vlm_kv_len = vlm_attention_mask.shape[1]
        assert state_len + latent_len + action_len == dit_len

        vlm_mask = torch.ones(batch_size, dit_len, vlm_kv_len, device=device, dtype=torch.bool)
        dit_self_mask = torch.zeros(batch_size, dit_len, dit_len, device=device, dtype=torch.bool)

        s_start, s_end = 0, state_len
        l_start, l_end = state_len, state_len + latent_len
        a_start, a_end = state_len + latent_len, dit_len

        if state_len > 0:
            dit_self_mask[:, s_start:s_end, s_start:s_end] = True
        if latent_len > 0:
            if state_len > 0:
                dit_self_mask[:, l_start:l_end, s_start:s_end] = True
            dit_self_mask[:, l_start:l_end, l_start:l_end] = True
        if action_len > 0:
            dit_self_mask[:, a_start:a_end, :] = True

        full_mask = torch.cat([vlm_mask, dit_self_mask], dim=-1)
        full_mask = full_mask.unsqueeze(1)
        float_mask = torch.zeros_like(full_mask, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~full_mask, float("-inf"))
        return float_mask

    def _prepare_dit_input(self, action_features, latent_features, state=None):
        parts = []
        if state is not None and self.state_encoder is not None:
            state_embed = self.state_encoder(state)
            parts.append(state_embed)
        parts.append(latent_features)
        parts.append(action_features)
        return torch.cat(parts, dim=1)

    def _extract_vlm_kv(self, vlm_past_key_values, start_idx, num_layers):
        selected_kv = []
        for i in range(start_idx, start_idx + num_layers):
            layer_cache = vlm_past_key_values[i]
            if isinstance(layer_cache, tuple):
                k, v = layer_cache
            elif hasattr(layer_cache, "key_cache") and hasattr(layer_cache, "value_cache"):
                k = layer_cache.key_cache
                v = layer_cache.value_cache
            elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
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

    def _get_vlm_kv_start_idx(self, vlm_past_key_values):
        vlm_num_layers = len(vlm_past_key_values)
        if self.vlm_kv_layer_offset is not None:
            vlm_kv_start_idx = vlm_num_layers - self.vlm_kv_layer_offset - self.dit_num_layers
            if vlm_kv_start_idx < 0:
                raise ValueError(
                    f"vlm_kv_layer_offset ({self.vlm_kv_layer_offset}) + dit_num_layers "
                    f"({self.dit_num_layers}) exceeds VLM layers ({vlm_num_layers})"
                )
        else:
            vlm_kv_start_idx = vlm_num_layers - self.dit_num_layers
            if vlm_kv_start_idx < 0:
                raise ValueError(f"dit_num_layers ({self.dit_num_layers}) cannot exceed VLM layers ({vlm_num_layers})")
        return vlm_kv_start_idx

    def forward(
        self,
        vlm_past_key_values: list,
        vlm_attention_mask: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        present_latents: torch.Tensor = None,
        future_latents: torch.Tensor = None,
        compute_action_loss_flags: torch.Tensor = None,
    ) -> dict:
        device = actions.device
        batch_size = actions.shape[0]

        # ── Flow-matching setup ────────────────────────────────────────────
        t_continuous = self.sample_time(batch_size, device, actions.dtype)
        t_discrete = (t_continuous * self.num_timestep_buckets).long()
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        noisy_trajectory = (1 - t_continuous[:, None, None]) * noise + t_continuous[:, None, None] * actions
        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_discrete)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # ── Compress present latents, then encode ─────────────────────────
        if self.latent_compressor is not None:
            compressed_present = self.latent_compressor(present_latents)  # [B, N//k, D]
        else:
            compressed_present = present_latents
        latent_features = self.latent_encoder(compressed_present)  # [B, N//k, hidden]
        latent_len = latent_features.shape[1]

        # ── Prepare DiT input: [state] + [latent] + [action] ──────────────
        hidden_states = self._prepare_dit_input(action_features, latent_features, state)
        dit_len = hidden_states.shape[1]
        state_len = 1 if (state is not None and self.state_encoder is not None) else 0
        action_len = action_features.shape[1]

        vlm_kv_start_idx = self._get_vlm_kv_start_idx(vlm_past_key_values)
        selected_vlm_kv = self._extract_vlm_kv(
            vlm_past_key_values, start_idx=vlm_kv_start_idx, num_layers=self.dit_num_layers
        )
        vlm_kv_len = selected_vlm_kv[0][0].shape[2]

        attention_mask = self._build_attention_mask(
            vlm_attention_mask,
            dit_len,
            device,
            vlm_kv_len=vlm_kv_len,
            state_len=state_len,
            latent_len=latent_len,
            action_len=action_len,
        )

        vlm_max_pos = vlm_attention_mask.sum(dim=1).max().item()
        dit_position_ids = (
            torch.arange(dit_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) + vlm_max_pos + 1
        )

        hidden_states = self.dit(
            hidden_states=hidden_states,
            vlm_past_key_values=selected_vlm_kv,
            t_continuous=t_continuous,
            dit_position_ids=dit_position_ids,
            attention_mask=attention_mask,
        )

        # ── Action loss ────────────────────────────────────────────────────
        action_hidden = hidden_states[:, -action_len:]
        pred_actions = self.action_decoder(action_hidden)
        sq_err = (pred_actions - velocity) ** 2

        if compute_action_loss_flags is not None:
            sample_gate = compute_action_loss_flags.float().unsqueeze(-1).unsqueeze(-1)
            if action_mask is not None:
                mask_f = action_mask.to(sq_err.dtype) * sample_gate
                denom = mask_f.sum().clamp_min(1.0)
                action_loss = (sq_err * mask_f).sum() / denom
            else:
                gated_sq_err = sq_err * sample_gate
                denom = (sample_gate.sum() * sq_err.shape[1] * sq_err.shape[2]).clamp_min(1.0)
                action_loss = gated_sq_err.sum() / denom
        else:
            if action_mask is not None:
                mask_f = action_mask.to(sq_err.dtype)
                denom = mask_f.sum().clamp_min(1.0)
                action_loss = (sq_err * mask_f).sum() / denom
            else:
                action_loss = sq_err.mean()

        # ── Latent prediction loss ─────────────────────────────────────────
        latent_hidden = hidden_states[:, state_len : state_len + latent_len]  # [B, N//k, hidden]
        pred_latents = self.latent_decoder(latent_hidden)  # [B, N//k, D]

        future_target = self.latent_compressor(future_latents)

        latent_loss = F.mse_loss(pred_latents, future_target.to(pred_latents.dtype))

        # ── Total loss ─────────────────────────────────────────────────────
        total_loss = self.action_loss_weight * action_loss + self.latent_loss_weight * latent_loss

        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "latent_loss": latent_loss,
        }

    @torch.no_grad()
    def predict_action(
        self,
        vlm_past_key_values: list,
        vlm_attention_mask: torch.Tensor,
        state: torch.Tensor = None,
        present_latents: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Inference: iterative denoising with prefix KV caching.
        LatentExpander is intentionally NOT called here — reconstructing the full-length
        latent sequence during deployment has no value and only wastes compute.
        """
        batch_size = vlm_attention_mask.shape[0]
        device = vlm_attention_mask.device

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        # ── Compress present latents ────────────────────
        if self.latent_compressor is not None and present_latents is not None:
            present_latents = self.latent_compressor(present_latents)  # [B, N//k, D]

        if present_latents is not None:
            latent_features = self.latent_encoder(present_latents)  # [B, N//k, hidden]
        else:
            latent_features = torch.zeros(batch_size, 1, self.action_hidden_dim, device=device, dtype=torch.float32)
        latent_len = latent_features.shape[1]

        state_len = 1 if (state is not None and self.state_encoder is not None) else 0
        action_len = self.action_horizon
        prefix_len = state_len + latent_len

        vlm_kv_start_idx = self._get_vlm_kv_start_idx(vlm_past_key_values)
        selected_vlm_kv = self._extract_vlm_kv(
            vlm_past_key_values, start_idx=vlm_kv_start_idx, num_layers=self.dit_num_layers
        )
        vlm_kv_len = selected_vlm_kv[0][0].shape[2]

        vlm_max_pos = vlm_attention_mask.sum(dim=1).max().item()
        dit_len = prefix_len + action_len

        # ── Phase 1: Prefill prefix (state + compressed latent) ───────────
        prefix_parts = []
        if state is not None and self.state_encoder is not None:
            state_embed = self.state_encoder(state)
            prefix_parts.append(state_embed)
        prefix_parts.append(latent_features)
        prefix_hidden = torch.cat(prefix_parts, dim=1)

        prefix_position_ids = (
            torch.arange(prefix_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            + vlm_max_pos
            + 1
        )

        prefix_attn_mask = torch.zeros(
            batch_size, 1, prefix_len, vlm_kv_len + prefix_len, device=device, dtype=torch.float32
        )

        prefix_kv_cache = self.dit.prefill_prefix(
            prefix_hidden_states=prefix_hidden,
            vlm_past_key_values=selected_vlm_kv,
            dit_position_ids=prefix_position_ids,
            attention_mask=prefix_attn_mask,
        )

        # ── Phase 2: Denoising loop (action tokens only) ──────────────────
        action_position_ids = (
            torch.arange(action_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            + vlm_max_pos
            + 1
            + prefix_len
        )

        action_attn_mask = torch.zeros(
            batch_size, 1, action_len, vlm_kv_len + prefix_len + action_len, device=device, dtype=torch.float32
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_continuous = torch.full(
                size=(batch_size,), fill_value=step / num_steps, device=device, dtype=torch.float32
            )
            t_discrete = (t_continuous * self.num_timestep_buckets).long()

            action_features = self.action_encoder(actions, t_discrete)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_len, dtype=torch.long, device=device) + prefix_len
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            hidden_states = self.dit(
                hidden_states=action_features,
                vlm_past_key_values=selected_vlm_kv,
                t_continuous=t_continuous,
                dit_position_ids=action_position_ids,
                attention_mask=action_attn_mask,
                prefix_kv_cache=prefix_kv_cache,
            )

            pred_velocity = self.action_decoder(hidden_states)
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_latent_action_model(config=None, latent_dim=None):
    """
    Factory: build KV shared flow-matching action head with compressed latent prediction.

    Args:
        config: Global config (expects config.framework.action_model namespace).
        latent_dim: Dimension of the latent features from the extractor.

    Returns:
        KVSharedLatentCompressedFlowmatchingActionHead: Initialized action head.
    """
    return KVSharedLatentCompressedFlowmatchingActionHead(
        global_config=config,
        latent_dim=latent_dim,
    )
