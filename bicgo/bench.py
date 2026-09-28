"""Throughput / memory benchmark for bicgo on a single accelerator.

Usage:
    python -m bicgo.bench --config configs/default.json [--iters 5]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .go_env import reset
from .model import build_model
from .ppo import compute_gae
from .selfplay import make_rollout
from .train import create_train_state, make_update


def _time(fn, iters: int):
    fn()
    jax.block_until_ready(None)
    t0 = time.time()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.time() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()

    key = jax.random.PRNGKey(0)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    rollout = make_rollout(model, cfg)
    update = make_update(model, cfg)

    state = reset(cfg.num_envs, cfg)
    t0 = time.time()
    tr, state, last_value, _ = rollout(ts.params, state, key)
    jax.block_until_ready(tr.value)
    print(f"compile rollout: {time.time()-t0:.1f}s")

    t_roll = _time(lambda: rollout(ts.params, state, key), args.iters)
    n = cfg.num_envs * cfg.rollout_len
    print(f"rollout : {t_roll*1000:8.1f} ms  {n/t_roll:12,.0f} env-steps/s")

    adv, tgt = compute_gae(
        tr.value, tr.reward, tr.done, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda
    )
    t0 = time.time()
    ts, metrics = update(ts, tr, adv, tgt, key)
    jax.block_until_ready(metrics["loss"])
    print(f"compile update : {time.time()-t0:.1f}s")
    t_upd = _time(lambda: update(ts, tr, adv, tgt, key), args.iters)
    print(
        f"update  : {t_upd*1000:8.1f} ms  {n*cfg.ppo.epochs/t_upd:12,.0f} sample-updates/s"
    )
    print(f"iteration: {t_roll+t_upd:6.2f}s  ({n} samples)")

    mem = jax.local_devices()[0].memory_stats()
    if mem:
        print(
            f"memory  : peak {mem.get('peak_bytes_in_use',0)/2**30:.1f} GiB "
            f"of {jax.local_devices()[0].memory_stats().get('bytes_limit',0)/2**30:.0f} GiB"
        )
    print(
        f"params  : {sum(x.size for x in jax.tree_util.tree_leaves(ts.params))/1e6:.2f}M"
    )


if __name__ == "__main__":
    main()
