"""ResNet -> ViT -> ResNet policy/value network.

The spatial matrix stays ``(S, S) = (N+1, N+1)`` end to end:

  stem conv -> [ResNet blocks] -> [ViT blocks over S*S tokens] -> [ResNet
  blocks] -> policy head (S*S cell logits + 1 pass logit) and a value MLP
  attached to the pooled features of the final (key) hidden layer.

The value head predicts the game outcome (win ``+1`` / loss ``-1``) from the
perspective of the player to move.

``dtype`` selects the activation/compute type (``bfloat16`` uses the A100
tensor cores for the conv/matmul heavy stages); parameters are kept in
float32.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import ModelConfig
from .features import encode, gather_board_logits


def compute_dtype(name: str):
    return jnp.bfloat16 if str(name).lower() in ("bfloat16", "bf16") else jnp.float32


class ResBlock(nn.Module):
    channels: int
    groups: int = 8
    dt: object = jnp.float32
    dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False, dtype=self.dt)(x)
        y = nn.GroupNorm(self.groups, dtype=self.dt)(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False, dtype=self.dt)(y)
        y = nn.GroupNorm(self.groups, dtype=self.dt)(y)
        y = nn.Dropout(self.dropout, deterministic=self.deterministic)(y)
        return nn.relu(x + y)


class ViTBlock(nn.Module):
    d_model: int
    heads: int
    mlp_ratio: int = 4
    dt: object = jnp.float32
    dropout: float = 0.0
    deterministic: bool = True
    attn_impl: str = "xla"

    @nn.compact
    def __call__(self, x, mask):
        b, t, _ = x.shape
        head_dim = self.d_model // self.heads
        y = nn.LayerNorm(dtype=self.dt)(x)
        qkv = nn.Dense(3 * self.d_model, name="qkv", dtype=self.dt)(y)
        qkv = qkv.reshape(b, t, 3, self.heads, head_dim)
        # jax.nn.dot_product_attention expects (B, T, heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        impl = None if self.attn_impl in (None, "", "xla") else self.attn_impl
        o = jax.nn.dot_product_attention(q, k, v, mask=mask, implementation=impl)
        o = o.reshape(b, t, self.d_model)
        o = nn.Dense(self.d_model, name="out", dtype=self.dt)(o)
        x = x + nn.Dropout(self.dropout, deterministic=self.deterministic)(o)
        z = nn.LayerNorm(dtype=self.dt)(x)
        z = nn.Dense(self.mlp_ratio * self.d_model, dtype=self.dt)(z)
        z = nn.gelu(z)
        z = nn.Dense(self.d_model, dtype=self.dt)(z)
        z = nn.Dropout(self.dropout, deterministic=self.deterministic)(z)
        return x + z


class ViTStack(nn.Module):
    d_model: int
    heads: int
    mlp_ratio: int
    depth: int
    dt: object = jnp.float32
    dropout: float = 0.0
    deterministic: bool = True
    attn_impl: str = "xla"

    @nn.compact
    def __call__(self, x, mask):
        for i in range(self.depth):
            x = ViTBlock(
                self.d_model, self.heads, self.mlp_ratio, self.dt,
                self.dropout, self.deterministic, self.attn_impl, name=f"vit{i}",
            )(x, mask)
        return x


class GoNet(nn.Module):
    cfg: ModelConfig
    board_size: int
    padded_size: int
    in_channels: int
    dt: object = jnp.float32

    @nn.compact
    def __call__(self, x, valid, deterministic: bool = True):
        c = self.cfg.channels
        s = self.padded_size
        batch = x.shape[0]
        dt = self.dt
        dp = self.cfg.dropout

        remat = self.cfg.remat
        Res = nn.remat(ResBlock) if remat else ResBlock
        ViT = nn.remat(ViTStack) if remat else ViTStack

        h = nn.Conv(c, (3, 3), padding="SAME", name="stem", dtype=dt)(x)
        h = nn.GroupNorm(self.cfg.group_norm_groups, name="stem_gn", dtype=dt)(h)
        h = nn.relu(h)
        for i in range(self.cfg.res_blocks_pre):
            h = Res(c, self.cfg.group_norm_groups, dt, dp, deterministic,
                    name=f"pre_res{i}")(h)

        tokens = h.reshape(batch, s * s, c)
        pos = self.param(
            "pos_emb", nn.initializers.normal(stddev=0.02), (1, s * s, c)
        )
        tokens = tokens + pos
        v = valid.reshape(batch, s * s).astype(bool)
        # Allow every token to attend to itself so no query row is ever fully
        # masked (cuDNN flash attention returns NaN for fully-masked rows).
        eye = jnp.eye(s * s, dtype=bool)[None]
        attn_mask = ((v[:, None, :] & v[:, :, None]) | eye)[:, None, :, :]
        tokens = ViT(c, self.cfg.vit_heads, self.cfg.vit_mlp_ratio, self.cfg.vit_depth,
                     dt, dp, deterministic, self.cfg.attn_impl)(tokens, attn_mask)
        h = tokens.reshape(batch, s, s, c)

        for i in range(self.cfg.res_blocks_post):
            h = Res(c, self.cfg.group_norm_groups, dt, dp, deterministic,
                    name=f"post_res{i}")(h)

        # value head attached to the key hidden layer (masked global pooling)
        vf = valid[..., None].astype(h.dtype)
        pooled = (h * vf).sum(axis=(1, 2)) / jnp.maximum(vf.sum(axis=(1, 2)), 1.0)
        vh = nn.Dense(self.cfg.value_hidden, name="value_fc1", dtype=jnp.float32)(pooled)
        vh = nn.relu(vh)
        vh = nn.Dense(1, name="value_fc2", dtype=jnp.float32)(vh)
        value = jnp.tanh(vh[..., 0])

        # policy head: per-cell logits + a pass logit from the pooled features
        cell = nn.Conv(1, (1, 1), name="policy_conv", dtype=jnp.float32)(h)[..., 0]
        cell = cell.reshape(batch, s * s)
        pass_logit = nn.Dense(1, name="policy_pass", dtype=jnp.float32)(pooled)
        logits = jnp.concatenate([cell, pass_logit], axis=-1)
        return logits, value


def build_model(cfg) -> GoNet:
    return GoNet(
        cfg=cfg.model,
        board_size=cfg.board_size,
        padded_size=cfg.padded_size,
        in_channels=cfg.in_channels,
        dt=compute_dtype(cfg.dtype),
    )


def forward(model, params, obs, valid, pad_r, pad_c, cfg, deterministic: bool = True):
    """Run the net and return board-coordinate logits ``(B, N*N+1)`` and value."""
    logits_pad, value = model.apply(
        {"params": params}, obs, valid, deterministic=deterministic
    )
    s = cfg.padded_size
    cell = logits_pad[:, : s * s].reshape(obs.shape[0], s, s)
    board = gather_board_logits(cell, pad_r, pad_c, cfg)
    logits = jnp.concatenate([board, logits_pad[:, -1:]], axis=-1)
    return logits, value


def masked_log_probs(logits, legal):
    return jax.nn.log_softmax(jnp.where(legal, logits, jnp.float32(-1e9)), axis=-1)
