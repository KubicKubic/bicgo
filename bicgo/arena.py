"""Fast batched game playing for evaluation and the Elo ladder.

``make_arena`` returns a cached factory ``get_match(n_games, opponent)`` whose
result is a jitted function ``match(params_a, params_b, key) -> win-rate`` for
player A.  ``opponent`` is one of ``policy`` (params_b), ``self`` (soft mirror
of A), ``random`` (uniform legal) or ``pass``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .features import encode
from .go_env import BLACK, WHITE, legal_mask, reset, score, step as env_step
from .model import compute_dtype, forward


def make_arena(model, cfg):
    dtype = compute_dtype(cfg.dtype)
    cache: dict = {}

    def get_match(n_games: int, opponent: str):
        key = (int(n_games), str(opponent))
        if key in cache:
            return cache[key]

        @jax.jit
        def match(params_a, params_b, rng):
            n = int(n_games)
            state = reset(n, cfg)
            a_colour = jnp.where(
                jnp.arange(n) % 2 == 0, jnp.int8(BLACK), jnp.int8(WHITE)
            )
            pad = jnp.zeros((n,), jnp.int32)

            def scan_step(carry, _):
                state, rng = carry
                k_act, rng = jax.random.split(rng)
                obs = encode(state, pad, pad, cfg, dtype)
                valid = obs[..., -1]
                logits_a, _ = forward(model, params_a, obs, valid, pad, pad, cfg)
                legal = legal_mask(
                    state.board, state.ko, state.to_play, cfg.env.allow_suicide
                )
                act_a = jax.random.categorical(
                    k_act, jnp.where(legal, logits_a, jnp.float32(-1e9))
                )
                is_a = state.to_play == a_colour

                if opponent in ("policy", "self"):
                    logits_b, _ = forward(model, params_b, obs, valid, pad, pad, cfg)
                    act_b = jax.random.categorical(
                        k_act, jnp.where(legal, logits_b, jnp.float32(-1e9))
                    )
                elif opponent == "random":
                    act_b = jax.random.categorical(
                        k_act, jnp.where(legal, 0.0, jnp.float32(-1e9))
                    )
                else:  # pass
                    act_b = jnp.full((n,), cfg.board_size ** 2, jnp.int32)
                action = jnp.where(is_a, act_a, act_b)

                nxt = env_step(state, action, cfg, score_now=False)
                nxt = nxt._replace(done=nxt.done | state.done)
                return (nxt, rng), None

            (final, _), _ = lax.scan(
                scan_step, (state, rng), None, length=cfg.ladder_max_moves
            )
            black_area, white_area, _ = score(final.board, cfg.env.komi)
            # Black wins iff black_area - white_area - komi > 0 (komi 7.5 -> no draws)
            black_wins = black_area - white_area - cfg.env.komi > 0
            a_is_black = a_colour == BLACK
            win_a = (a_is_black == black_wins).astype(jnp.float32)
            return win_a.mean()

        cache[key] = match
        return match

    def play(params_a, params_b, rng, n_games: int, opponent: str) -> float:
        return float(get_match(n_games, opponent)(params_a, params_b, rng))

    return play


def make_value_eval(model, cfg):
    """Evaluate the value head against the *real* terminal of self-play games.

    Plays ``cfg.eval_games`` self-play games for ``cfg.value_eval_moves`` steps
    so most games actually finish (two passes -> area result, hard cap -> draw).
    Only positions from finished games are scored, using the exact outcome
    (``+1`` win / ``-1`` loss / ``0`` draw) for the player to move.  Reports
    overall metrics plus an opening/mid/end game breakdown, because a Go value
    head is genuinely uncertain in the opening and sharp in the endgame.

      * ``ev_mean``/``ev_actual``  mean predicted vs realised score
      * ``value_mse``              Brier/MSE (0 perfect, 1 equals trivial v=0)
      * ``value_acc``              sign agreement
      * ``ev_explained``           1 - MSE/Var(z) (1 perfect, 0 trivial)
      * ``value_acc_open/mid/end`` accuracy by move phase
    """
    n = int(cfg.eval_games)
    dtype = compute_dtype(cfg.dtype)
    steps = int(cfg.value_eval_moves)
    mid, end = 100, 300

    @jax.jit
    def run(params, rng):
        state = reset(n, cfg)
        pad = jnp.zeros((n,), jnp.int32)

        def scan_step(carry, _):
            state, rng = carry
            k_act, rng = jax.random.split(rng)
            obs = encode(state, pad, pad, cfg, dtype)
            logits, value = forward(model, params, obs, obs[..., -1], pad, pad, cfg)
            legal = legal_mask(
                state.board, state.ko, state.to_play, cfg.env.allow_suicide
            )
            action = jax.random.categorical(
                k_act, jnp.where(legal, logits, jnp.float32(-1e9))
            )
            rec = {
                "value": value,
                "player": state.to_play.astype(jnp.float32),
                "active": (~state.done).astype(jnp.float32),
            }
            nxt = env_step(state, action, cfg, score_now=False)
            nxt = nxt._replace(done=nxt.done | state.done)
            return (nxt, rng), rec

        (final, _), rec = lax.scan(scan_step, (state, rng), None, length=steps)

        black_area, white_area, _ = score(final.board, cfg.env.komi)
        black_wins = black_area - white_area - cfg.env.komi > 0
        two_pass = final.passes >= 2  # else hard-cap draw -> z = 0
        winner = jnp.where(black_wins, jnp.float32(BLACK), jnp.float32(WHITE))
        # only positions from games that actually finished, and z = 0 on draws
        mask = rec["active"] * final.done[None, :].astype(jnp.float32)
        won = (rec["player"] == winner[None, :]).astype(jnp.float32)
        z = jnp.where(two_pass[None, :], jnp.where(won > 0, 1.0, -1.0), 0.0) * mask
        v = rec["value"] * mask

        def stats(m):
            c = jnp.maximum(m.sum(), 1.0)
            vv, zz = v * m, z * m
            ev_actual = zz.sum() / c
            mse = ((vv - zz) ** 2).sum() / c
            acc = (((vv > 0) == (zz > 0)).astype(jnp.float32) * m).sum() / c
            var = ((zz - ev_actual) ** 2).sum() / c
            expl = jnp.where(var > 1e-4, 1.0 - mse / jnp.maximum(var, 1e-6), 0.0)
            return vv.sum() / c, ev_actual, mse, acc, expl

        moves = jnp.arange(steps)[:, None] * jnp.ones((1, n))
        open_m = mask * (moves < mid)
        mid_m = mask * (moves >= mid) * (moves < end)
        end_m = mask * (moves >= end)
        ev_mean, ev_actual, mse, acc, expl = stats(mask)
        _, _, _, acc_open, _ = stats(open_m)
        _, _, _, acc_mid, _ = stats(mid_m)
        _, _, _, acc_end, _ = stats(end_m)
        return {
            "ev_mean": ev_mean,
            "ev_actual": ev_actual,
            "value_mse": mse,
            "value_acc": acc,
            "ev_explained": expl,
            "value_acc_open": acc_open,
            "value_acc_mid": acc_mid,
            "value_acc_end": acc_end,
            "ev_games_done": final.done.mean(),
            "ev_positions": mask.sum(),
        }

    return run
