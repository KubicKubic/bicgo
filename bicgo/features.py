"""Board encoding with randomized 19x19 -> 20x20 padding.

The network always sees a square ``(S, S)`` matrix with ``S = N + 1``.  The
19x19 board is dropped into one of the four corners of the 20x20 canvas; the
chosen corner is randomized every step and acts as a cheap data augmentation
(the ViT positional embedding therefore sees the board at different offsets).
The final plane is a validity mask telling the network which cells are real.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .go_env import BLACK, WHITE


def random_pad_offsets(key: jnp.ndarray, batch: int, pad: int):
    """Return ``(pad_r, pad_c)`` int32 arrays of shape ``(batch,)`` in ``0..pad``."""
    if pad == 0:
        z = jnp.zeros((batch,), dtype=jnp.int32)
        return z, z
    k1, k2 = jax.random.split(key)
    pad_r = jax.random.randint(k1, (batch,), 0, pad + 1, dtype=jnp.int32)
    pad_c = jax.random.randint(k2, (batch,), 0, pad + 1, dtype=jnp.int32)
    return pad_r, pad_c


def make_pad_offsets(cfg, key: jnp.ndarray, batch: int):
    """Random corner offsets when ``cfg.randomize_pad`` else a fixed corner."""
    if getattr(cfg, "randomize_pad", True):
        return random_pad_offsets(key, batch, cfg.pad)
    z = jnp.zeros((batch,), dtype=jnp.int32)
    return z, z


def encode(state, pad_r, pad_c, cfg, dtype=jnp.float32) -> jnp.ndarray:
    """Encode a batched :class:`GoState` into ``(B, S, S, C)`` planes.

    ``C = 2 * history + 3``: own/opponent history planes, a side-to-move plane,
    a ko plane and the validity mask.
    """
    n = cfg.board_size
    s = n + cfg.pad
    h = cfg.env.history if hasattr(cfg, "env") else cfg.history
    board = state.board
    batch = board.shape[0]
    tp = state.to_play.astype(jnp.int32)
    opp = 3 - tp

    hist = state.hist
    own = (hist == tp[:, None, None, None]).astype(jnp.float32)
    foe = (hist == opp[:, None, None, None]).astype(jnp.float32)
    planes = jnp.concatenate([own, foe], axis=1)  # (B, 2H, N, N)

    side = jnp.broadcast_to(
        (tp == BLACK).astype(jnp.float32)[:, None, None, None], (batch, 1, n, n)
    )
    grid = jnp.arange(n * n).reshape(n, n)
    ko_plane = (grid[None] == state.ko[:, None, None]).astype(jnp.float32)[:, None]
    planes = jnp.concatenate([planes, side, ko_plane], axis=1)  # (B, C-1, N, N)

    planes = jnp.transpose(planes, (0, 2, 3, 1))  # (B, N, N, C-1)

    def place(p, r, c):
        canvas = jnp.zeros((s, s, p.shape[-1]), dtype=jnp.float32)
        return lax.dynamic_update_slice(canvas, p, (r, c, 0))

    canvas = jax.vmap(place)(planes, pad_r, pad_c)  # (B, S, S, C-1)

    def valid_one(r, c):
        v = jnp.zeros((s, s), dtype=jnp.float32)
        return lax.dynamic_update_slice(v, jnp.ones((n, n), dtype=jnp.float32), (r, c))

    valid = jax.vmap(valid_one)(pad_r, pad_c)[..., None]  # (B, S, S, 1)
    obs = jnp.concatenate([canvas, valid], axis=-1)
    return obs.astype(dtype)


def gather_board_logits(padded_logits: jnp.ndarray, pad_r, pad_c, cfg) -> jnp.ndarray:
    """Gather the ``N*N`` board logits out of a ``(B, S, S)`` padded map."""
    n = cfg.board_size
    rows = pad_r[:, None] + jnp.arange(n, dtype=jnp.int32)[None, :]
    cols = pad_c[:, None] + jnp.arange(n, dtype=jnp.int32)[None, :]

    def one(logits, rr, cc):
        return logits[rr[:, None], cc[None, :]]

    return jax.vmap(one)(padded_logits, rows, cols).reshape(padded_logits.shape[0], n * n)
