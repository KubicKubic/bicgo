"""Leela Zero network ported to JAX.

The LZ net is a v1 AlphaZero-style ResNet (18 input planes -> residual tower ->
policy/value heads).  This module loads the official text weight files
(zero.sjeng.org) and runs the network batched in JAX, so it can be used as a
strong, fast opponent for evaluation and as a possible initialisation.

Weight order and head math follow leela-zero ``src/Network.cpp`` exactly:

  * conv weights are ``[out][in][kh][kw]``
  * every conv is followed by ``relu(scale * (conv_with_bias(x) - mean))`` with
    ``scale = 1/sqrt(var + 1e-5)`` and no activation on the second residual conv
    before the skip-add (the ``relu`` wraps the sum)
  * policy: 1x1 conv -> per-channel BN+relu -> FC(362) -> softmax
  * value:  1x1 conv -> per-channel BN+relu -> FC(256)+relu -> FC(1)
            -> winrate = (1 + tanh(.)) / 2   (v1: player-to-move)
"""
from __future__ import annotations

import gzip
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-5


def load_lz_text(path: str | Path):
    """Parse an LZ text weight file into named numpy arrays."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.read().split("\n")
    counts = [len(l.split()) for l in lines]
    version = int(lines[0].split()[0])
    c = counts[2]  # input conv bias line -> filters
    # locate start of blocks: line0 version, then 4 input lines
    def vec(i):
        return np.asarray(lines[i].split(), dtype=np.float32)

    def conv(i, out_ch, in_ch, k=3):
        w = vec(i).reshape(out_ch, in_ch, k, k)
        return w

    def fc(i, out_ch, in_ch):
        return vec(i).reshape(out_ch, in_ch)

    w = {}
    w["version"] = version
    w["input_w"] = conv(1, c, 18)
    w["input_b"] = vec(2)
    w["input_m"] = vec(3)
    w["input_s"] = 1.0 / np.sqrt(vec(4) + EPS)
    off = 5
    blocks = []
    nb = (len(counts) - (1 + 4 + 14)) // 8
    for _ in range(nb):
        blocks.append({
            "w1": conv(off, c, c), "b1": vec(off + 1),
            "m1": vec(off + 2), "s1": 1.0 / np.sqrt(vec(off + 3) + EPS),
            "w2": conv(off + 4, c, c), "b2": vec(off + 5),
            "m2": vec(off + 6), "s2": 1.0 / np.sqrt(vec(off + 7) + EPS),
        })
        off += 8
    w["blocks"] = blocks
    w["pol_w"] = conv(off, 2, c, 1)
    w["pol_b"] = vec(off + 1)
    w["pol_m"] = vec(off + 2)
    w["pol_s"] = 1.0 / np.sqrt(vec(off + 3) + EPS)
    w["ip_pol_w"] = fc(off + 4, 362, 2 * 361)
    w["ip_pol_b"] = vec(off + 5)
    w["val_w"] = conv(off + 6, 1, c, 1)
    w["val_b"] = vec(off + 7)
    w["val_m"] = vec(off + 8)
    w["val_s"] = 1.0 / np.sqrt(vec(off + 9) + EPS)
    w["ip1_val_w"] = fc(off + 10, 256, 361)
    w["ip1_val_b"] = vec(off + 11)
    w["ip2_val_w"] = vec(off + 12).reshape(1, 256)
    w["ip2_val_b"] = vec(off + 13)
    return w


def _bn(x, mean, scale):
    """relu(scale * (x - mean)) with channel-broadcast params (NCHW)."""
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return jax.nn.relu(scale.reshape(shape) * (x - mean.reshape(shape)))


def _conv(x, w, b, k):
    y = jax.lax.conv_general_dilated(
        x, w, (1, 1), "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + b.reshape(1, -1, 1, 1)


def lz_forward(params, x, softmax_temp: float = 1.0):
    """x: (B,18,19,19) float32 -> (policy_probs (B,362), winrate (B,))."""
    h = _bn(_conv(x, params["input_w"], params["input_b"], 3),
            params["input_m"], params["input_s"])
    for blk in params["blocks"]:
        a = _bn(_conv(h, blk["w1"], blk["b1"], 3), blk["m1"], blk["s1"])
        b = _conv(a, blk["w2"], blk["b2"], 3)
        shape = (1, -1, 1, 1)
        h = jax.nn.relu(blk["s2"].reshape(shape) * (b - blk["m2"].reshape(shape)) + h)
    p = _bn(_conv(h, params["pol_w"], params["pol_b"], 1), params["pol_m"], params["pol_s"])
    p = p.reshape(x.shape[0], -1)
    logits = p @ params["ip_pol_w"].T + params["ip_pol_b"]
    logits = logits / softmax_temp
    probs = jax.nn.softmax(logits, axis=-1)
    v = _bn(_conv(h, params["val_w"], params["val_b"], 1), params["val_m"], params["val_s"])
    v = v.reshape(x.shape[0], -1)
    v = jax.nn.relu(v @ params["ip1_val_w"].T + params["ip1_val_b"])
    out = (v @ params["ip2_val_w"].T + params["ip2_val_b"])[..., 0]
    winrate = (1.0 + jnp.tanh(out)) / 2.0
    return probs, winrate


def make_lz_params(w: dict):
    """Convert numpy weights to jnp arrays (a pytree)."""
    def cvt(blk):
        return {k: jnp.asarray(v) for k, v in blk.items()}
    out = {k: (jnp.asarray(v) if not isinstance(v, list) else [cvt(b) for b in v])
           for k, v in w.items() if k not in ("version",)}
    return out


# ---------------------------------------------------------------------------
# features (identity symmetry), matching Network::gather_features
# ---------------------------------------------------------------------------
def lz_features(board_hist, to_play):
    """board_hist: (B,H,19,19) int8 with hist[:,0]=current; to_play: (B,) int.

    Returns (B,18,19,19): 8 own history planes, 8 opponent planes, then the
    black-to-move and white-to-move indicator planes.
    """
    B, H, N, _ = board_hist.shape
    H = min(H, 8)
    tp = to_play.astype(jnp.int32)
    opp = 3 - tp
    planes = []
    for h in range(8):
        if h < H:
            bh = board_hist[:, h]
        else:
            bh = jnp.zeros((B, N, N), dtype=board_hist.dtype)
        planes.append((bh == tp[:, None, None]).astype(jnp.float32))
    for h in range(8):
        if h < H:
            bh = board_hist[:, h]
        else:
            bh = jnp.zeros((B, N, N), dtype=board_hist.dtype)
        planes.append((bh == opp[:, None, None]).astype(jnp.float32))
    black_move = (tp == 1).astype(jnp.float32)[:, None, None] * jnp.ones((B, N, N))
    white_move = (tp == 2).astype(jnp.float32)[:, None, None] * jnp.ones((B, N, N))
    planes.append(black_move)
    planes.append(white_move)
    return jnp.stack(planes, axis=1)  # (B,18,N,N)


def make_lz_opponent(net_path, cfg, temperature: float = 1.0):
    """Return jitted (params, policy_fn, value_fn) for batched LZ play."""
    params = make_lz_params(load_lz_text(net_path))

    @jax.jit
    def forward(x):
        return lz_forward(params, x, temperature)

    def policy_fn(state):
        x = lz_features(state.hist, state.to_play)
        probs, wr = forward(x)
        return probs, wr

    return params, policy_fn
