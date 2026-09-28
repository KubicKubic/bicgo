"""Highly parallel evaluation of bicgo against an external GTP engine.

GNU Go / Pachi cannot be ported to JAX (heuristic C engines), but we can run
many engine processes **concurrently** while batching the bicgo network forward
across all in-flight games.  Concretely, for ``--games K`` we:

  * start K GTP engine subprocesses (one per game, colour-balanced),
  * query all engine `genmove`s **in parallel threads** for the plies where the
    engine moves, and
  * evaluate bicgo's move for every game where it moves with a **single batched
    JAX forward pass**,

so the wall-clock cost is ~ (engine ply) + (one batched forward) per move, not
K times that.  Legality/scoring use the bicgo env (Chinese area, komi 7.5).

    python -m bicgo.gtp_parallel --checkpoint runs/.../ckpt_004700 \
        --engine "/usr/games/gnugo --mode gtp --chinese-rules --komi 7.5" \
        --games 32
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import go_env as E
from .config import TrainConfig
from .gtp_match import Engine, _gtp_to_index, make_fns
from .sgf import index_to_vertex


def run_games(policy_fn, step_fn, legal_fn, params, cfg, engine_cmd, n_games,
              verbose=False):
    size = cfg.board_size
    engines = [Engine(engine_cmd) for _ in range(n_games)]
    ours_black = np.array([i % 2 == 0 for i in range(n_games)])
    our_colour = np.where(ours_black, E.BLACK, E.WHITE).astype(np.int32)
    for e in engines:
        for c in ("boardsize %d" % size, "komi %.1f" % cfg.env.komi, "clear_board"):
            e.cmd(c)

    state = E.reset(n_games, cfg)
    done = np.zeros(n_games, bool)
    result = np.full(n_games, -1.0)
    moves = np.zeros(n_games, np.int64)
    pool = ThreadPoolExecutor(max_workers=n_games)

    try:
        while not done.all() and moves.min() < cfg.env.max_moves:
            active = (~done) & (moves < cfg.env.max_moves)
            turn = np.asarray(state.to_play)
            ours = active & (turn == our_colour)
            eng = active & (turn != our_colour)
            legal_all = np.asarray(legal_fn(state))  # (K, A)

            # --- engine moves in parallel threads ---
            def ask(i):
                for _ in range(4):
                    resp = engines[i].cmd(f"genmove {'B' if turn[i] == E.BLACK else 'W'}")
                    idx, kind = _gtp_to_index(resp, size)
                    if kind != "move":
                        return i, idx, kind
                    if legal_all[i, idx]:
                        return i, idx, "move"
                    engines[i].cmd("undo")
                return i, size * size, "pass"

            eng_action = np.full(n_games, size * size, np.int64)
            eng_kind = ["none"] * n_games
            futs = [pool.submit(ask, int(i)) for i in np.where(eng)[0]]
            for f in futs:
                i, idx, kind = f.result()
                eng_kind[i] = kind
                if kind == "resign":
                    result[i] = 1.0
                    done[i] = True
                    eng_action[i] = size * size
                else:
                    eng_action[i] = idx

            # --- one batched bicgo forward for all our games ---
            model_action = np.asarray(policy_fn(params, state))
            action = np.where(ours, model_action, eng_action)
            action = np.where(active, action, size * size).astype(np.int32)

            # send our played moves back to the engines (parallel)
            send = [pool.submit(
                engines[int(i)].cmd,
                f"play {'B' if turn[i] == E.BLACK else 'W'} {index_to_vertex(int(model_action[i]), size)}",
            ) for i in np.where(ours)[0]]
            for f in send:
                f.result()

            state = step_fn(state, jnp.asarray(action))
            moves[active] += 1

            # --- resolve games that just ended ---
            just = np.asarray(state.done) & ~done
            if just.any():
                passes = np.asarray(state.passes)
                ba, wa, winner = E.score(state.board, cfg.env.komi)
                ba, wa, winner = np.asarray(ba), np.asarray(wa), np.asarray(winner)
                for i in np.where(just)[0]:
                    if passes[i] >= 2:
                        result[i] = 1.0 if winner[i] == our_colour[i] else 0.0
                    else:
                        result[i] = 0.5  # hard-cap draw
                    done[i] = True
                    if verbose:
                        print(f"  game {i}: {'WIN' if result[i]==1 else ('DRAW' if result[i]==0.5 else 'LOSS')}"
                              f" moves={moves[i]} area b/w={ba[i]}/{wa[i]}", flush=True)
    finally:
        for e in engines:
            e.close()
        pool.shutdown(wait=False)

    played = int((result >= 0).sum())
    wins = float(result[result >= 0].sum())
    return wins, played, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base = TrainConfig.from_json("configs/default.json")
    cfg = TrainConfig.from_dict(
        {**base.to_dict(),
         "env": {**base.to_dict()["env"], "board_size": args.size,
                 "max_moves": 4 * args.size * args.size}}
    )
    from .train import load_params

    from .model import build_model

    key = jax.random.PRNGKey(0)
    model = build_model(cfg)
    params = load_params(Path(args.checkpoint), model, cfg, key)
    policy_fn, step_fn, legal_fn = make_fns(model, cfg)

    import time
    t0 = time.time()
    wins, played, _ = run_games(policy_fn, step_fn, legal_fn, params, cfg,
                                args.engine, args.games, args.verbose)
    dt = time.time() - t0
    print(f"bicgo vs engine: {wins:.0f}/{played} = {wins/max(played,1):.3f} "
          f"({played} games in {dt:.0f}s, {played/max(dt,1e-9):.2f} games/s)",
          flush=True)


if __name__ == "__main__":
    main()
