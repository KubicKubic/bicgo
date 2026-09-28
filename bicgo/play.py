"""Play against a trained bicgo checkpoint in the terminal.

Examples:
    python -m bicgo.play --checkpoint runs/bicgo/ckpt_final --color b
    python -m bicgo.play --checkpoint runs/bicgo/ckpt_final --color w --sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from . import go_env as E
from .config import TrainConfig
from .features import encode
from .model import build_model, forward
from .sgf import index_to_vertex, moves_to_sgf, vertex_to_index

STONES = {E.EMPTY: ".", E.BLACK: "X", E.WHITE: "O"}
COLS = "ABCDEFGHJKLMNOPQRST"


def print_board(board):
    n = board.shape[0]
    header = "   " + " ".join(COLS[c] for c in range(n))
    print(header)
    for r in range(n):
        print(f"{n - r:2d} " + " ".join(STONES[int(board[r, c])] for c in range(n))
              + f" {n - r}")
    print(header)


def _model_move(model, params, cfg, state, sample, key):
    pad = jnp.zeros((1,), jnp.int32)
    obs = encode(state, pad, pad, cfg)
    logits, _ = forward(model, params, obs, obs[..., -1], pad, pad, cfg)
    legal = E.legal_mask(state.board, state.ko, state.to_play, cfg.env.allow_suicide)
    masked = jnp.where(legal, logits, jnp.float32(-1e9))
    if sample:
        key, k = jax.random.split(key)
        return int(jax.random.categorical(k, masked)[0]), key
    return int(jnp.argmax(masked[0])), key


def main():
    p = argparse.ArgumentParser(description="Play Go against a bicgo checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--color", choices=["b", "w"], default="b")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--sample", action="store_true", help="sample instead of argmax")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sgf-out", default=None, help="write the finished game to SGF")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    cfg_path = ckpt / "config.json"
    cfg = TrainConfig.from_json(cfg_path) if cfg_path.exists() else TrainConfig()
    d = cfg.to_dict()
    d["num_envs"] = 1
    if args.size:
        d["env"] = {**d["env"], "board_size": args.size,
                    "max_moves": 4 * args.size * args.size}
    cfg = TrainConfig.from_dict(d)

    from .train import load_params

    key = jax.random.PRNGKey(args.seed)
    model = build_model(cfg)
    params = load_params(ckpt, model, cfg, key)
    state = E.reset(1, cfg)
    human = E.BLACK if args.color == "b" else E.WHITE
    moves: list[int] = []

    print(f"board {cfg.board_size}x{cfg.board_size}, komi {cfg.env.komi}, "
          f"you are {'black' if human == E.BLACK else 'white'}")
    print("enter a coordinate (e.g. D4), 'pass', or 'q' to quit\n")

    while not bool(state.done[0]):
        print_board(state.board[0])
        player = int(state.to_play[0])
        if player == human:
            raw = input("your move> ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            action = vertex_to_index(raw, cfg.board_size)
            if action is None:
                print("invalid coordinate")
                continue
            legal = E.legal_mask(state.board, state.ko, state.to_play,
                                 cfg.env.allow_suicide)
            if not bool(legal[0, action]):
                print("illegal move")
                continue
        else:
            action, key = _model_move(model, params, cfg, state, args.sample, key)
            print(f"model plays {index_to_vertex(action, cfg.board_size)}")
        moves.append(action)
        state = E.step(state, jnp.array([action], jnp.int32), cfg)

    if bool(state.done[0]):
        print_board(state.board[0])
        winner = "black" if int(state.winner[0]) == E.BLACK else "white"
        print(f"game over: {winner} wins "
              f"(black {int(state.black_area[0])} / white {int(state.white_area[0])})")
    sgf = moves_to_sgf(moves, cfg.board_size, cfg.env.komi)
    print(sgf)
    if args.sgf_out:
        Path(args.sgf_out).write_text(sgf + "\n")
        print(f"wrote {args.sgf_out}")


if __name__ == "__main__":
    main()
