"""Batched self-play rollout.

Every parallel game plays exactly ``cfg.rollout_len`` moves per iteration. A
game that terminates is reset in place immediately, so no parallelism is
wasted; unfinished games bootstrap their value at the end of the rollout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .features import encode, gather_board_logits, make_pad_offsets
from .go_env import (
    BLACK,
    DRAW,
    WHITE,
    finalize_scores,
    legal_mask,
    reset,
    step as env_step,
    terminal_reward,
)
from .model import compute_dtype, masked_log_probs
from .ppo import Transition


def auto_reset(state, done, cfg):
    """Replace finished games with a fresh board (per-field where)."""
    fresh = reset(state.board.shape[0], cfg)

    def select(old, new):
        mask = done.reshape(done.shape + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(select, state, fresh)


def make_rollout(model, cfg):
    """Build a jitted rollout closure returning a :class:`Transition`."""
    budget = cfg.rollout_len
    batch = cfg.num_envs
    dtype = compute_dtype(cfg.dtype)

    def rollout(params, state, key):
        def scan_step(carry, _):
            state, rng = carry
            k_pad, k_act, rng = jax.random.split(rng, 3)

            pad_r, pad_c = make_pad_offsets(cfg, k_pad, batch)
            obs = encode(state, pad_r, pad_c, cfg, dtype)
            valid = obs[..., -1]
            logits_pad, value = model.apply({"params": params}, obs, valid)

            s = cfg.padded_size
            cell = logits_pad[:, : s * s].reshape(batch, s, s)
            board = gather_board_logits(cell, pad_r, pad_c, cfg)
            logits = jnp.concatenate([board, logits_pad[:, -1:]], axis=-1)

            legal = legal_mask(
                state.board, state.ko, state.to_play, cfg.env.allow_suicide
            )
            logp_all = masked_log_probs(logits, legal)
            action = jax.random.categorical(
                k_act, jnp.where(legal, logits, jnp.float32(-1e9))
            )
            logp = jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0]

            prev = state
            nxt = env_step(state, action, cfg, score_now=False)
            just_finished = nxt.done & ~prev.done
            nxt = finalize_scores(nxt, just_finished, cfg.env.komi)
            reward = terminal_reward(nxt, prev)
            done = nxt.done
            state_next = auto_reset(nxt, done, cfg)

            transition = Transition(
                obs=obs.astype(jnp.uint8),
                valid=valid.astype(jnp.uint8),
                legal=legal,
                pad_r=pad_r,
                pad_c=pad_c,
                action=action,
                log_prob=logp,
                value=value,
                reward=reward,
                done=done.astype(jnp.float32),
            )
            stats = {
                "finished": just_finished.astype(jnp.float32),
                "passes": (action >= cfg.board_size ** 2).astype(jnp.float32),
                "game_len": jnp.where(
                    just_finished, prev.move_num.astype(jnp.float32), 0.0
                ),
                "black_win": ((nxt.winner == BLACK) & just_finished).astype(jnp.float32),
                "white_win": ((nxt.winner == WHITE) & just_finished).astype(jnp.float32),
                "draw": ((nxt.winner == DRAW) & just_finished).astype(jnp.float32),
            }
            return (state_next, rng), (transition, stats)

        (final_state, _), (transitions, stats) = lax.scan(
            scan_step, (state, key), None, length=budget
        )

        k_pad, _ = jax.random.split(key)
        pad_r, pad_c = make_pad_offsets(cfg, k_pad, batch)
        obs = encode(final_state, pad_r, pad_c, cfg, dtype)
        _, last_value = model.apply({"params": params}, obs, obs[..., -1])

        finished = stats["finished"].sum()
        aggregate = {
            "games_finished": finished,
            "finish_per_game": finished / batch,
            "pass_frac": stats["passes"].mean(),
            "avg_game_len": stats["game_len"].sum() / jnp.maximum(finished, 1.0),
            "black_win_frac": stats["black_win"].sum() / jnp.maximum(finished, 1.0),
            "draw_frac": stats["draw"].sum() / jnp.maximum(finished, 1.0),
        }
        return transitions, final_state, last_value, aggregate

    return jax.jit(rollout)
