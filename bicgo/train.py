"""Training entry point for bicgo."""
from __future__ import annotations

import os

# JAX reserves 75% of GPU memory by default; bicgo's default config peaks well
# below that, so cap the reservation to keep the shared A100 usable by others.
# Set XLA_PYTHON_CLIENT_MEM_FRACTION externally to override.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state as _ts

from .config import TrainConfig
from .go_env import reset
from .model import build_model
from .ppo import compute_gae, ppo_loss
from .selfplay import make_rollout

METRIC_KEYS = (
    "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
)


class TrainState(_ts.TrainState):
    pass


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def _steps_per_iter(cfg: TrainConfig) -> int:
    n = cfg.num_envs * cfg.rollout_len
    mb = max(1, min(n, cfg.ppo.minibatch_size))
    num_mb = max(1, n // mb)
    return cfg.ppo.epochs * num_mb


def entropy_coef_at(iteration: int, cfg: TrainConfig) -> float:
    start = cfg.ppo.entropy_coef_start
    start = cfg.ppo.entropy_coef if start is None else start
    end = cfg.ppo.entropy_coef_end
    end = cfg.ppo.entropy_coef if end is None else end
    total = cfg.ppo.entropy_schedule_iters or cfg.iterations
    frac = 1.0 if total <= 0 else min(1.0, iteration / total)
    return float(start + (end - start) * frac)


def create_train_state(model, cfg: TrainConfig, key):
    dummy = jnp.zeros(
        (1, cfg.padded_size, cfg.padded_size, cfg.in_channels), dtype=jnp.float32
    )
    valid = jnp.ones((1, cfg.padded_size, cfg.padded_size), dtype=jnp.float32)
    params = model.init(key, dummy, valid)["params"]

    total_iter = cfg.ppo.lr_total_iters or cfg.iterations
    warm = cfg.ppo.lr_warmup_iters * _steps_per_iter(cfg)
    decay = max(warm + 1, total_iter * _steps_per_iter(cfg))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.ppo.lr,
        warmup_steps=warm,
        decay_steps=decay,
        end_value=cfg.ppo.lr_min,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.ppo.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ---------------------------------------------------------------------------
# PPO update (with entropy schedule + KL early stop)
# ---------------------------------------------------------------------------
def make_update(model, cfg: TrainConfig):
    def loss_fn(params, batch, adv, tgt, entropy_coef):
        return ppo_loss(model, params, batch, adv, tgt, cfg, entropy_coef)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(train_state, transitions, advantages, targets, key, entropy_coef):
        n = advantages.shape[0] * advantages.shape[1]
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape((n,) + x.shape[2:]), transitions
        )
        adv = advantages.reshape(-1)
        tgt = targets.reshape(-1)
        if cfg.ppo.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        mb_size = max(1, min(n, cfg.ppo.minibatch_size))
        num_mb = max(1, n // mb_size)
        used = num_mb * mb_size
        epoch_keys = jax.random.split(key, cfg.ppo.epochs)

        def run_epoch(ts, ek):
            perm = jax.random.permutation(ek, n)[:used]
            batches = perm.reshape(num_mb, mb_size)

            def mb(ts, idx):
                batch = jax.tree_util.tree_map(lambda x: x[idx], flat)
                (loss, metrics), grads = grad_fn(
                    ts.params, batch, adv[idx], tgt[idx], entropy_coef
                )
                ts = ts.apply_gradients(grads=grads)
                return ts, metrics

            ts, metrics = jax.lax.scan(mb, ts, batches)
            return ts, jax.tree_util.tree_map(lambda x: x.mean(), metrics)

        zero = {k: jnp.float32(0.0) for k in METRIC_KEYS}

        def cond(carry):
            _, i, stop, _, _ = carry
            return (i < cfg.ppo.epochs) & (~stop)

        def body(carry):
            ts, i, stop, acc, cnt = carry
            ts, m = run_epoch(ts, epoch_keys[i])
            acc = jax.tree_util.tree_map(lambda a, b: a + b, acc, m)
            return ts, i + 1, m["approx_kl"] > cfg.ppo.kl_target, acc, cnt + 1.0

        ts, _, _, acc, cnt = jax.lax.while_loop(
            cond, body, (train_state, jnp.int32(0), jnp.array(False), zero, jnp.float32(0.0))
        )
        metrics = jax.tree_util.tree_map(lambda a: a / jnp.maximum(cnt, 1.0), acc)
        metrics["epochs_run"] = cnt
        return ts, metrics

    return jax.jit(update)


# ---------------------------------------------------------------------------
# evaluation (arena)
# ---------------------------------------------------------------------------
def make_evaluator(model, cfg: TrainConfig):
    from .arena import make_arena

    play = make_arena(model, cfg)

    def evaluate(params, key):
        opponent = cfg.eval_opponent
        params_b = params if opponent in ("policy", "self") else params
        return play(params, params_b, key, cfg.eval_games, opponent)

    return evaluate


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, train_state, cfg: TrainConfig, iteration: int | None = None):
    from flax import serialization

    path.mkdir(parents=True, exist_ok=True)
    (path / "params.msgpack").write_bytes(serialization.to_bytes(train_state.params))
    (path / "train_state.msgpack").write_bytes(serialization.to_bytes(train_state))
    (path / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (path / "meta.json").write_text(json.dumps({"iteration": iteration}))


def load_params(path, model, cfg: TrainConfig, key):
    """Load parameters saved by :func:`save_checkpoint`."""
    from flax import serialization

    dummy = jnp.zeros(
        (1, cfg.padded_size, cfg.padded_size, cfg.in_channels), dtype=jnp.float32
    )
    valid = jnp.ones((1, cfg.padded_size, cfg.padded_size), dtype=jnp.float32)
    template = model.init(key, dummy, valid)["params"]
    blob = (Path(path) / "params.msgpack").read_bytes()
    return serialization.from_bytes(template, blob)


def load_train_state(path, model, cfg: TrainConfig, key):
    """Restore a full TrainState (params + optimizer moments + step)."""
    from flax import serialization

    template = create_train_state(model, cfg, key)
    blob = (Path(path) / "train_state.msgpack").read_bytes()
    meta = json.loads((Path(path) / "meta.json").read_text())
    return serialization.from_bytes(template, blob), int(meta.get("iteration") or 0)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def _setup_runtime(cfg: TrainConfig, out: Path):
    if cfg.comp_cache:
        cache = out / "jax_cache"
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    devices = jax.devices()
    if 0 <= cfg.device < len(devices):
        jax.config.update("jax_default_device", devices[cfg.device])


def train(cfg: TrainConfig, smoke: bool = False):
    if smoke:
        cfg = TrainConfig.from_dict(
            {**cfg.to_dict(), "iterations": 2, "num_envs": 8,
             "rollout_len": 16, "eval_games": 4}
        )

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    _setup_runtime(cfg, out)

    key = jax.random.PRNGKey(cfg.ppo.seed)
    model = build_model(cfg)
    key, k_init = jax.random.split(key)

    start_iter = 0
    if cfg.resume:
        train_state, last_iter = load_train_state(cfg.resume, model, cfg, k_init)
        start_iter = last_iter
        print(f"[bicgo] resumed from {cfg.resume} at iteration {last_iter}")
    else:
        train_state = create_train_state(model, cfg, k_init)

    rollout = make_rollout(model, cfg)
    update = make_update(model, cfg)
    evaluate = make_evaluator(model, cfg)

    ladder = None
    if cfg.ladder:
        from .arena import make_arena
        from .ladder import EloLadder

        ladder = EloLadder(cfg, make_arena(model, cfg), train_state.params, 0)

    value_eval = None
    if cfg.eval_value:
        from .arena import make_value_eval

        value_eval = make_value_eval(model, cfg)

    metrics_fp = open(out / "metrics.jsonl", "a") if cfg.log_jsonl else None
    eval_fp = open(out / "eval.jsonl", "a") if cfg.log_jsonl else None
    ladder_fp = open(out / "ladder.jsonl", "a") if cfg.log_jsonl else None
    value_fp = open(out / "value.jsonl", "a") if cfg.log_jsonl else None

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
    print(f"[bicgo] params={n_params/1e6:.2f}M board={cfg.board_size} "
          f"pad={cfg.pad} envs={cfg.num_envs} budget={cfg.rollout_len} "
          f"dtype={cfg.dtype} ladder={cfg.ladder}", flush=True)

    state = reset(cfg.num_envs, cfg)
    last_good = (train_state.params, train_state.opt_state)
    t0 = time.time()
    for it in range(start_iter + 1, cfg.iterations + 1):
        key, kr, ku = jax.random.split(key, 3)
        transitions, state, last_value, agg = rollout(train_state.params, state, kr)
        advantages, targets = compute_gae(
            transitions.value, transitions.reward, transitions.done, last_value,
            cfg.ppo.gamma, cfg.ppo.gae_lambda,
        )
        ec = entropy_coef_at(it, cfg)
        train_state, metrics = update(
            train_state, transitions, advantages, targets, ku, jnp.float32(ec)
        )

        finite = bool(np.isfinite(np.asarray(jax.device_get(metrics["loss"]))))
        if not finite and last_good is not None:
            # roll back the diverged update and continue from the last good step
            train_state = train_state.replace(
                params=last_good[0], opt_state=last_good[1]
            )
            print(f"[{it:06d}] NaN detected -> rolled back to last good step",
                  flush=True)
        else:
            last_good = (train_state.params, train_state.opt_state)

        if it % cfg.log_every == 0:
            m = jax.device_get(metrics)
            dt = (time.time() - t0) / cfg.log_every
            t0 = time.time()
            rec = {k: float(m[k]) for k in METRIC_KEYS}
            agg_d = {k: float(v) for k, v in jax.device_get(agg).items()}
            rec.update(iteration=it, entropy_coef=ec, seconds=dt,
                       epochs_run=float(m["epochs_run"]), **agg_d)
            print(
                f"[{it:06d}] loss={rec['loss']:.4f} pi={rec['policy_loss']:.4f} "
                f"vf={rec['value_loss']:.4f} ent={rec['entropy']:.4f} "
                f"kl={rec['approx_kl']:.4f} clip={rec['clip_frac']:.3f} "
                f"ec={ec:.4f} ep={float(m['epochs_run']):.1f} "
                f"fin/g={agg_d['finish_per_game']:.2f} pass={agg_d['pass_frac']:.3f} "
                f"len={agg_d['avg_game_len']:.0f} bw={agg_d['black_win_frac']:.2f} "
                f"draw={agg_d['draw_frac']:.2f} {dt:.2f}s/it",
                flush=True,
            )
            if metrics_fp:
                metrics_fp.write(json.dumps(rec) + "\n")
                metrics_fp.flush()

        if cfg.eval_every and it % cfg.eval_every == 0:
            key, ke = jax.random.split(key)
            wr = float(evaluate(train_state.params, ke))
            print(f"[{it:06d}] eval win-rate vs {cfg.eval_opponent} = {wr:.3f}", flush=True)
            if eval_fp:
                eval_fp.write(json.dumps(
                    {"iteration": it, "opponent": cfg.eval_opponent, "win_rate": wr}
                ) + "\n")
                eval_fp.flush()
            if ladder is not None:
                key, kl = jax.random.split(key)
                rating, results = ladder.evaluate_and_update(train_state.params, it, kl)
                detail = " ".join(f"{k}:{v:.2f}" for k, v in sorted(results.items()))
                print(f"[{it:06d}] elo = {rating:.1f}   ({detail})", flush=True)
                if ladder_fp:
                    ladder_fp.write(json.dumps(
                        {"iteration": it, "elo": rating,
                         "opponents": {str(k): v for k, v in results.items()}}
                    ) + "\n")
                    ladder_fp.flush()
            if value_eval is not None:
                key, kv = jax.random.split(key)
                vd = {k: float(v) for k, v in jax.device_get(value_eval(train_state.params, kv)).items()}
                print(
                    f"[{it:06d}] value EV: ev={vd['ev_mean']:.3f} "
                    f"actual={vd['ev_actual']:.3f} mse={vd['value_mse']:.3f} "
                    f"acc={vd['value_acc']:.3f} expl={vd['ev_explained']:.3f} "
                    f"acc[open/mid/end]={vd['value_acc_open']:.2f}/"
                    f"{vd['value_acc_mid']:.2f}/{vd['value_acc_end']:.2f} "
                    f"done={vd['ev_games_done']:.2f}",
                    flush=True,
                )
                if value_fp:
                    value_fp.write(json.dumps({"iteration": it, **vd}) + "\n")
                    value_fp.flush()

        if cfg.ckpt_every and it % cfg.ckpt_every == 0:
            save_checkpoint(out / f"ckpt_{it:06d}", train_state, cfg, it)

    save_checkpoint(out / "ckpt_final", train_state, cfg, cfg.iterations)
    for fp in (metrics_fp, eval_fp, ladder_fp, value_fp):
        if fp:
            fp.close()
    print("[bicgo] done", flush=True)


def main():
    p = argparse.ArgumentParser(description="Train bicgo (pure PPO Go)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--rollout-len", type=int, default=None)
    p.add_argument("--board-size", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke and not args.config:
        smoke_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
        if smoke_path.exists():
            args.config = str(smoke_path)

    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    d = cfg.to_dict()
    if args.out:
        d["out_dir"] = args.out
    if args.resume:
        d["resume"] = args.resume
    if args.iterations is not None:
        d["iterations"] = args.iterations
    if args.num_envs is not None:
        d["num_envs"] = args.num_envs
    if args.rollout_len is not None:
        d["rollout_len"] = args.rollout_len
    if args.board_size is not None:
        d["env"] = {**d["env"], "board_size": args.board_size,
                    "max_moves": 4 * args.board_size * args.board_size}
    cfg = TrainConfig.from_dict(d)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
