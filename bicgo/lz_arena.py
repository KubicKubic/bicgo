"""Evaluate bicgo against the JAX-ported Leela Zero net, fully batched.

No subprocesses: bicgo and the LZ net run in the same JIT, so ``--games 128``
takes seconds instead of minutes.

    python -m bicgo.lz_arena --checkpoint runs/.../ckpt_004700 \
        --lz-weights third_party/lz/lz15x192.gz --games 128
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import go_env as E
from .config import TrainConfig
from .features import encode
from .lz import lz_features, lz_forward, load_lz_text, make_lz_params
from .model import build_model, forward


def make_match(model, lz_params, cfg, n_games, temperature=1.0):
    @jax.jit
    def match(params, key):
        state = E.reset(n_games, cfg)
        a_colour = jnp.where(jnp.arange(n_games) % 2 == 0, jnp.int8(E.BLACK),
                             jnp.int8(E.WHITE))
        pad = jnp.zeros((n_games,), jnp.int32)

        def step(carry, _):
            state, rng = carry
            k_a, k_b, rng = jax.random.split(rng, 3)
            legal = E.legal_mask(state.board, state.ko, state.to_play,
                                 cfg.env.allow_suicide)
            # bicgo policy
            obs = encode(state, pad, pad, cfg)
            logits, _ = forward(model, params, obs, obs[..., -1], pad, pad, cfg)
            act_a = jax.random.categorical(
                k_a, jnp.where(legal, logits, jnp.float32(-1e9)))
            # Leela Zero policy (batched)
            x = lz_features(state.hist, state.to_play)
            probs, _ = lz_forward(lz_params, x, temperature)
            probs = probs * legal.astype(probs.dtype)
            probs = probs / jnp.maximum(probs.sum(-1, keepdims=True), 1e-9)
            act_b = jax.random.categorical(k_b, jnp.log(probs + 1e-9))
            is_a = state.to_play == a_colour
            action = jnp.where(is_a, act_a, act_b)
            nxt = E.step(state, action, cfg, score_now=False)
            nxt = nxt._replace(done=nxt.done | state.done)
            return (nxt, rng), None

        (final, _), _ = jax.lax.scan(
            step, (state, key), None, length=cfg.env.max_moves)
        black_wins = jnp.asarray(E.score(final.board, cfg.env.komi)[2]) == E.BLACK
        a_is_black = a_colour == E.BLACK
        win_a = (a_is_black == black_wins).astype(jnp.float32)
        # hard-cap draws count as half
        draw = final.passes < 2
        return jnp.where(draw, 0.5, win_a).mean()

    return match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lz-weights", default="third_party/lz/lz15x192.gz")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    cfg = TrainConfig.from_json("configs/default.json")
    from .train import load_params

    key = jax.random.PRNGKey(0)
    model = build_model(cfg)
    params = load_params(Path(args.checkpoint), model, cfg, key)
    lz_params = make_lz_params(load_lz_text(args.lz_weights))
    match = make_match(model, lz_params, cfg, args.games, args.temperature)

    for n in (args.games,):
        t0 = time.time()
        wr = float(match(params, key))
        print(f"bicgo vs LZ(15x192, policy-only) win rate: {wr:.3f} "
              f"({n} games in {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
