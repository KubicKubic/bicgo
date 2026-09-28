# bicgo — pure-PPO JAX trainer for 19×19 Go

A self-contained, `jit`/`vmap`-friendly Go environment and a pure PPO learner
written in JAX/Flax. No MCTS, no search: a policy/value network is trained
directly with PPO from self-play, with **terminal-only reward**.

```
ResNet  ->  ViT  ->  ResNet          (matrix stays 20×20 the whole way)
                     └─ value MLP head on the key hidden layer
```

## Rules (aligned with HullQin)

The environment reproduces the rules used by HullQin's weiqi game
(`game.hullqin.cn`, webpack chunk `wq`, modules `9568`/`8601`):

| aspect        | behaviour                                                            |
|---------------|----------------------------------------------------------------------|
| board         | 19×19 (9/13 also supported), `0` empty, `1` black, `2` white         |
| first move    | black                                                                |
| captures      | groups with no liberties are removed                                 |
| ko            | **simple ko** (single-stone immediate recapture is illegal)          |
| suicide       | illegal                                                              |
| pass          | flat index `N*N` (= 361)                                             |
| end of game   | **two consecutive passes** -> area score                             |
| hard cap      | `env.max_moves` (default `4*N*N`); reaching it is a **draw** (reward 0) |
| scoring       | **Chinese area scoring**, komi **7.5** (`黑贴3又3/4子`)               |
| winner        | black iff `black_area - komi > white_area`                           |

The rollout budget (`rollout_len`) **never truncates a game**. Games live
across rollouts: at the end of a budget the in-flight games are value
bootstrapped and their state is carried into the next rollout; only a real
terminal (two passes, or the long hard cap) ends a game and triggers an
in-place reset. The hard cap is deliberately long so it is rarely reached; a
cap ending counts as a draw for both players.

Two deliberate decisions on top of the reference implementation:

* HullQin's UI has **no automatic win/loss** (the loser must resign) and offers
  a non-standard *"reject pass"* button instead of ending on two passes.  For a
  trainable RL environment we use the classic two-consecutive-pass termination
  and decide the result with Chinese area scoring + komi 7.5 — the same komi the
  HullQin scoring aid documents.
* HullQin's scoring *aid* uses a heuristic dead-stone estimator that its own UI
  warns "needs manual verification".  We use exact area scoring instead (all
  stones on the board are counted as alive), which is the well-defined classic
  rule and a stable reward for self-play.

Everything is validated against an independent NumPy implementation
(`bicgo/reference.py`) over random playouts, including capture, suicide, ko,
two-pass termination and scoring (`tests/test_go_rules.py`).

## Network

* **Input**: 19×19 board padded to **20×20** with one extra row/column placed at
  a **random corner** each step (data augmentation), plus a validity plane.
  Planes = `2*history` (own/opponent stones) `+ side-to-move + ko + valid`
  (`history=8` → 19 planes).
* **Stem** 3×3 conv → `GroupNorm` → ReLU.
* **ResNet** stage (`res_blocks_pre` residual blocks, 3×3, same padding).
* **ViT** stage: each of the 400 cells is a token; learned positional
  embeddings; `vit_depth` transformer blocks with attention masked to valid
  cells. Spatial size is preserved (tokens are reshaped back to 20×20).
* **ResNet** stage (`res_blocks_post` blocks).
* **Policy head**: 1×1 conv → 400 cell logits + a pass logit from the pooled
  features; the 361 board logits are gathered out of the padded map.
* **Value head**: an MLP attached to the pooled features of the final (key)
  hidden layer; `tanh` output = expected result for the player to move.

## PPO

* `gamma = 1.0`, `gae_lambda = 0.9`.
* **Terminal-only reward**: `+1` for the winner, `-1` for the loser, `0`
  elsewhere.
* Clipped surrogate + value MSE + entropy bonus, grad-norm clipping, AdamW.
* Value is from the perspective of the player to move; because the players
  alternate, the effective GAE discount is `-gamma`, so the recursion is
  `A_t = δ_t − γλ·A_{t+1}` (`bicgo/ppo.py`). This is covered by a unit test.
* **Entropy schedule**: the entropy bonus starts high for exploration and
  decays linearly (`ppo.entropy_coef_start` → `entropy_coef_end` over
  `entropy_schedule_iters`); it is passed as a traced scalar so changing it
  never recompiles.
* **KL early stop**: if an epoch's mean `approx_kl` exceeds `ppo.kl_target`,
  the remaining epochs of that update are skipped.
* **LR schedule**: linear warmup then cosine decay to `ppo.lr_min`
  (`optax.warmup_cosine_decay_schedule`).
* **Dropout**: `model.dropout` (>0) is active during the update and disabled
  for rollouts/eval.

## Evaluation, Elo ladder & logging

* `eval_opponent` ∈ `random | self | pass | policy`, played batched and
  colour-balanced on the accelerator (`bicgo/arena.py`).
* **Elo ladder** (`bicgo/ladder.py`): every `eval_every` iterations the current
  model plays `ladder_games` games against a sample of `ladder_opponents` past
  snapshots; an Elo rating (K = `ladder_k`, anchored at 0 for the initial
  network) is updated and logged, giving a strength curve to watch during long
  runs.
* Per-iteration metrics, eval results and ladder ratings are appended as JSON
  lines to `out_dir/metrics.jsonl`, `eval.jsonl`, `ladder.jsonl`.
* Checkpoints contain `params.msgpack`, the full `train_state.msgpack`
  (optimizer moments) and `meta.json`; resume with `--resume <ckpt_dir>`.
* A JAX persistent compilation cache is written under `out_dir/jax_cache`
  (`comp_cache`).

## Rollout / parallelism

* `num_envs` parallel games, each playing exactly `rollout_len` moves per
  iteration (a fixed step budget).
* A game that terminates (two passes, or the long hard cap) is **reset in
  place immediately**, so no parallelism is wasted.
* Games still running at the end of the budget are **not truncated**: they are
  **value bootstrapped** and their exact state is handed to the next rollout.
  Terminal steps use the `done` mask to stop GAE.
* The metrics log tracks `finish_per_game`, `pass_frac`, `avg_game_len`,
  `black_win_frac` and `draw_frac` so terminal health is observable; `draw_frac`
  is the fraction of games that ended at the hard cap.

## Performance on an A100

Measured on a single `A100-SXM4-80GB` with the default config
(`num_envs=256`, `rollout_len=128`, 32768 samples/iteration, `epochs=4`,
`minibatch_size=1024`, `dtype=bfloat16`, `remat=false`, 3.26M params):

| stage                        | throughput                |
|------------------------------|---------------------------|
| self-play rollout            | ~12,300 env-steps/s       |
| PPO update                   | ~4,500 sample-updates/s   |
| whole iteration              | ~29 s                     |
| sampled SM utilisation       | steady ~100 % (no swings) |
| peak device memory           | ~35 GiB                   |

Notes / knobs:

* `dtype=bfloat16` uses the tensor cores (Flax keeps parameters in float32) and
  is the single biggest win; `dtype=float32` is ~1.5x slower.
* `remat=true` trades ~30% speed for memory; it is only needed if you enlarge
  the model or the minibatch.
* The update is compute-bound: throughput per sample is ~constant across
  `num_envs`/`minibatch_size`, so scale work per iteration with
  `num_envs`/`rollout_len`, not with a larger minibatch.
* JAX reserves 75% of GPU memory by default. The launcher caps it with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` (48 GiB) so the shared card stays usable;
  override the env var to change it.
* The Go rules use `while_loop` flood fills, but they are cheap (~2 ms per
  256-game step when jitted) relative to the network; avoid calling env
  functions eagerly (un-jitted) in a loop.

## Usage

```bash
PY=/mnt/pfs/guoyuchong/guoyuchong/.venv-gnn-jax/bin/python
cd bicgo

# fast smoke checks (uses configs/smoke.json: 9x9, tiny net, 4 envs)
scripts/smoke.sh                 # pytest tests/test_smoke.py + one tiny train run
SMOKE_FULL=1 scripts/smoke.sh    # also run the full test suite

# full run with the default config
$PY -m bicgo.train --config configs/default.json

# override common knobs
$PY -m bicgo.train --config configs/default.json \
    --num-envs 256 --rollout-len 256 --iterations 20000 --out runs/bicgo_a100

# resume from a checkpoint
$PY -m bicgo.train --config runs/bicgo_a100/config.json \
    --resume runs/bicgo_a100/ckpt_001000 --out runs/bicgo_a100

# play against a checkpoint (human vs model, exports SGF)
$PY -m bicgo.play --checkpoint runs/bicgo_a100/ckpt_final --color b --sgf-out game.sgf

# throughput / memory benchmark
$PY -m bicgo.bench --config configs/default.json

# tests
$PY -m pytest tests/ -q
```

## Smoke tests

`tests/test_smoke.py` is a fast end-to-end suite (no long training) that
exercises, in one process:

1. config JSON round-trip and derived sizes;
2. env reset / legal mask / step / two-pass termination / deferred scoring /
   terminal reward;
3. in-place auto-reset of finished games;
4. feature encoding, 20x20 padding validity and a model forward pass;
5. batched rollout (obs/log-prob/value shapes) + GAE + a PPO update that
   actually moves the parameters with finite metrics;
6. evaluation vs the random opponent;
7. checkpoint save/load round-trip;
8. a complete one-iteration `train()` run (rollout + update + eval + checkpoint).

`tests/test_go_rules.py` additionally cross-checks the JAX rules against an
independent NumPy implementation over random playouts (capture, suicide, ko,
two-pass end, area scoring), and `tests/test_model_and_ppo.py` checks shapes and
the GAE recursion.

Checkpoints are written to `out_dir/ckpt_*/params.msgpack` (flax serialization)
alongside a copy of the config.

## Layout

```
bicgo/
  bicgo/
    config.py      typed dataclass config (JSON round-trip)
    go_env.py      jitted/vmapped Go environment (rules, ko, scoring)
    features.py    plane encoding + randomized 20x20 padding + logit gather
    model.py       ResNet->ViT->ResNet + value MLP + policy head
    ppo.py         GAE and PPO loss
    selfplay.py    jitted batched rollout with in-place auto-reset
    arena.py       fast batched match play (policy / self / random / pass)
    ladder.py      Elo ladder over parameter snapshots
    sgf.py         minimal SGF import/export
    play.py        human-vs-model terminal CLI
    train.py       training loop, schedules, eval, ladder, logging, resume
    bench.py       throughput / memory benchmark
    reference.py   independent NumPy rules oracle
  configs/{default,smoke}.json
  scripts/{train,smoke}.sh
  tests/          20 tests: rules oracle, model/PPO, end-to-end smoke
```
