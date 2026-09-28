"""Play bicgo against an external GTP engine and report the win rate.

The bicgo environment is the source of truth for legality and scoring (Chinese
area, komi 7.5); every engine move is sent to our env and every bicgo move is
sent back to the engine with ``play`` so both boards stay in sync.

    python -m bicgo.gtp_match --checkpoint runs/.../ckpt_004700 \
        --engine "/usr/games/gnugo --mode gtp --chinese-rules --komi 7.5" \
        --games 10

Supports GNU Go and Pachi (both speak GTP). Env step / legality / policy are
jitted and scoring is deferred to game end, so a move costs a few ms.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

from . import go_env as E
from .config import TrainConfig
from .features import encode
from .model import build_model, forward
from .sgf import index_to_vertex, vertex_to_index


class Engine:
    def __init__(self, cmd: str):
        self.proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def cmd(self, line: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            resp = self.proc.stdout.readline()
            if not resp:
                raise RuntimeError("engine died")
            resp = resp.rstrip("\n")
            if resp.startswith("=") or resp.startswith("?"):
                out.append(resp[1:].strip())
                return " ".join(out)
            out.append(resp.strip())

    def close(self):
        try:
            self.cmd("quit")
        except Exception:
            pass
        self.proc.terminate()


def _gtp_to_index(v: str, size: int):
    v = v.strip().upper()
    if v.startswith("PASS"):
        return size * size, "pass"
    if v.startswith("RESIGN"):
        return None, "resign"
    idx = vertex_to_index(v, size)
    if idx is None:
        raise ValueError(f"bad gtp vertex {v!r}")
    return idx, "move"


def make_fns(model, cfg):
    """Return jitted (policy_fn, step_fn, legal_fn) for this config."""
    def act(params, state):
        b = state.board.shape[0]
        pad = jnp.zeros((b,), jnp.int32)
        obs = encode(state, pad, pad, cfg)
        logits, _ = forward(model, params, obs, obs[..., -1], pad, pad, cfg)
        legal = E.legal_mask(state.board, state.ko, state.to_play, cfg.env.allow_suicide)
        # returns (B,) actions; batch=1 callers take int(...)
        return jnp.argmax(jnp.where(legal, logits, jnp.float32(-1e9)), axis=-1)

    policy_fn = jax.jit(act)
    step_fn = jax.jit(lambda s, a: E.step(s, a, cfg, score_now=False))
    legal_fn = jax.jit(
        lambda s: E.legal_mask(s.board, s.ko, s.to_play, cfg.env.allow_suicide)
    )
    return policy_fn, step_fn, legal_fn


def play_game(policy_fn, step_fn, legal_fn, params, cfg, engine_cmd,
              size, ours_is_black, verbose=False):
    state = E.reset(1, cfg)
    engine = Engine(engine_cmd)
    try:
        for c in ("boardsize %d" % size, "komi %.1f" % cfg.env.komi, "clear_board"):
            engine.cmd(c)
        our_colour = E.BLACK if ours_is_black else E.WHITE
        moves = 0
        while not bool(state.done[0]) and moves < cfg.env.max_moves:
            turn = int(state.to_play[0])
            gtp = "B" if turn == E.BLACK else "W"
            ours = turn == our_colour
            if ours:
                idx = int(policy_fn(params, state))
                vertex = index_to_vertex(idx, size)
                resp = engine.cmd(f"play {gtp} {vertex}")
                if resp.startswith("?"):
                    raise RuntimeError(f"engine rejected our move {vertex}: {resp}")
            else:
                idx, kind = None, "pass"
                for attempt in range(4):
                    resp = engine.cmd(f"genmove {gtp}")
                    idx, kind = _gtp_to_index(resp, size)
                    if kind != "move":
                        break
                    if bool(legal_fn(state)[0, idx]):
                        break
                    engine.cmd("undo")
                    if attempt == 3:
                        idx, kind = size * size, "pass"
                if kind == "resign":
                    return 1.0, "resign"
                if verbose:
                    print(f"  engine {gtp} {index_to_vertex(idx, size)}")
            state = step_fn(state, jnp.array([idx], jnp.int32))
            moves += 1
        if bool(state.done[0]):
            if int(state.passes[0]) >= 2:
                _, _, winner = E.score(state.board, cfg.env.komi)
                return (1.0 if int(winner[0]) == our_colour else 0.0), "score"
            return 0.5, "draw"
        return 0.5, "max_moves"
    finally:
        engine.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--games", type=int, default=10)
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

    key = jax.random.PRNGKey(0)
    model = build_model(cfg)
    params = load_params(Path(args.checkpoint), model, cfg, key)
    policy_fn, step_fn, legal_fn = make_fns(model, cfg)

    wins = 0.0
    played = 0
    for g in range(args.games):
        ours_black = g % 2 == 0
        try:
            score, how = play_game(policy_fn, step_fn, legal_fn, params, cfg,
                                   args.engine, args.size, ours_black, args.verbose)
        except Exception as e:
            print(f"game {g+1}: ERROR {e}", flush=True)
            continue
        played += 1
        wins += score
        col = "black" if ours_black else "white"
        res = {1.0: "WIN", 0.0: "LOSS", 0.5: "DRAW"}[score]
        print(f"game {g+1:2d}: bicgo({col}) {res} ({how})  running={wins/played:.3f}",
              flush=True)
    print(f"\nbicgo win rate vs engine: {wins}/{played} = {wins/max(played,1):.3f}")


if __name__ == "__main__":
    main()
