"""Fast end-to-end smoke tests.

Uses ``configs/smoke.json`` (9x9 board, tiny network, 4 envs, 8-step rollout) so
the whole file compiles and runs quickly. Run with::

    pytest tests/test_smoke.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bicgo import go_env as E
from bicgo.config import EnvConfig, TrainConfig
from bicgo.features import encode, random_pad_offsets
from bicgo.model import build_model, forward
from bicgo.ppo import compute_gae
from bicgo.selfplay import auto_reset, make_rollout
from bicgo.train import (
    create_train_state,
    load_params,
    load_train_state,
    make_evaluator,
    make_update,
    save_checkpoint,
    train,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.json"


def _cfg(**overrides) -> TrainConfig:
    d = json.loads(SMOKE.read_text())
    d.update(overrides)
    return TrainConfig.from_dict(d)


def test_config_roundtrip():
    cfg = TrainConfig.from_json(SMOKE)
    again = TrainConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert cfg.dtype == "bfloat16"
    assert cfg.in_channels == 2 * cfg.env.history + 3
    assert cfg.num_actions == cfg.board_size ** 2 + 1


def test_env_reset_legal_step_finalize():
    cfg = _cfg()
    state = E.reset(cfg.num_envs, cfg)
    assert state.board.shape == (cfg.num_envs, 9, 9)
    assert bool((state.to_play == E.BLACK).all())
    legal = E.legal_mask(state.board, state.ko, state.to_play)
    assert legal.shape == (cfg.num_envs, 82)
    assert bool(legal[:, 81].all())  # pass is always legal
    # black and white each pass -> game ends, white wins the empty board (komi)
    prev = state
    for _ in range(2):
        prev = state
        state = E.step(state, jnp.full((cfg.num_envs,), 81, jnp.int32), cfg,
                       score_now=False)
    just = state.done & ~prev.done
    assert bool(just.all())
    state = E.finalize_scores(state, just, cfg.env.komi)
    assert bool((state.winner == E.WHITE).all())
    # White made the second (terminal) pass and wins the empty board
    assert bool((prev.to_play == E.WHITE).all())
    reward = E.terminal_reward(state, prev)
    assert float(reward[0]) == 1.0


def test_auto_reset():
    cfg = _cfg()
    state = E.reset(cfg.num_envs, cfg)
    state = state._replace(done=jnp.ones((cfg.num_envs,), bool))
    state = state._replace(board=state.board.at[:, 0, 0].set(1))
    fresh = auto_reset(state, state.done, cfg)
    assert not bool(fresh.done.any())
    assert int(fresh.board.sum()) == 0
    assert bool((fresh.to_play == E.BLACK).all())


def test_features_and_forward():
    cfg = _cfg()
    state = E.reset(cfg.num_envs, cfg)
    key = jax.random.PRNGKey(0)
    pad_r, pad_c = random_pad_offsets(key, cfg.num_envs, cfg.pad)
    obs = encode(state, pad_r, pad_c, cfg)
    s = cfg.padded_size
    assert obs.shape == (cfg.num_envs, s, s, cfg.in_channels)
    valid = np.asarray(obs[..., -1])
    for b in range(cfg.num_envs):
        assert valid[b].sum() == 81
    model = build_model(cfg)
    params = model.init(key, obs, obs[..., -1])["params"]
    logits, value = forward(model, params, obs, obs[..., -1], pad_r, pad_c, cfg)
    assert logits.shape == (cfg.num_envs, cfg.num_actions)
    assert value.shape == (cfg.num_envs,)
    assert np.isfinite(np.asarray(logits)).all()
    assert np.all(np.abs(np.asarray(value)) <= 1.0)


def test_rollout_gae_update():
    cfg = _cfg()
    key = jax.random.PRNGKey(1)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    rollout = make_rollout(model, cfg)
    update = make_update(model, cfg)

    state = E.reset(cfg.num_envs, cfg)
    tr, state, last_value, _ = rollout(ts.params, state, key)
    assert tr.obs.shape == (cfg.rollout_len, cfg.num_envs, cfg.padded_size,
                            cfg.padded_size, cfg.in_channels)
    assert np.isfinite(np.asarray(tr.log_prob)).all()
    assert np.isfinite(np.asarray(tr.value)).all()

    adv, tgt = compute_gae(
        tr.value, tr.reward, tr.done, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda
    )
    assert np.isfinite(np.asarray(adv)).all()
    before = jax.tree_util.tree_leaves(ts.params)[0].copy()
    ts2, metrics = update(ts, tr, adv, tgt, key, jnp.float32(0.01))
    after = jax.tree_util.tree_leaves(ts2.params)[0]
    assert not np.allclose(np.asarray(before), np.asarray(after))
    for v in jax.tree_util.tree_leaves(metrics):
        assert np.isfinite(np.asarray(v))


def test_evaluator():
    cfg = _cfg()
    key = jax.random.PRNGKey(2)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    evaluate = make_evaluator(model, cfg)
    wr = float(evaluate(ts.params, key))
    assert 0.0 <= wr <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    key = jax.random.PRNGKey(3)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, ts, cfg)
    assert (ckpt / "params.msgpack").exists()
    loaded = load_params(ckpt, model, cfg, key)
    for a, b in zip(
        jax.tree_util.tree_leaves(ts.params), jax.tree_util.tree_leaves(loaded)
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_train_smoke(tmp_path):
    cfg = _cfg(out_dir=str(tmp_path / "run"))
    train(cfg)  # one iteration, includes eval + checkpoint
    assert (tmp_path / "run" / "config.json").exists()
    assert (tmp_path / "run" / "ckpt_final" / "params.msgpack").exists()


def test_arena_and_ladder():
    from bicgo.arena import make_arena
    from bicgo.ladder import EloLadder

    cfg = _cfg()
    key = jax.random.PRNGKey(4)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    play = make_arena(model, cfg)
    r = play(ts.params, ts.params, key, 8, "pass")
    assert r == 1.0  # always-pass opponent loses
    assert 0.0 <= play(ts.params, ts.params, key, 8, "policy") <= 1.0
    ladder = EloLadder(cfg, play, ts.params, 0)
    rating, results = ladder.evaluate_and_update(ts.params, 1, key)
    assert np.isfinite(rating) and len(results) == 1 and len(ladder.entries) == 2


def test_rollout_actions_are_legal():
    """Every sampled action must be legal under its own stored mask."""
    cfg = _cfg()
    key = jax.random.PRNGKey(7)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    rollout = make_rollout(model, cfg)
    state = E.reset(cfg.num_envs, cfg)
    tr, _, _, _ = rollout(ts.params, state, key)
    taken = jnp.take_along_axis(tr.legal, tr.action[..., None], axis=-1)[..., 0]
    assert bool(taken.all())
    assert int(tr.action.max()) <= cfg.num_actions - 1
    assert int(tr.action.min()) >= 0


def test_history_orientation():
    """hist[:,0] is the current board, hist[:,1] the previous one."""
    cfg = EnvConfig(board_size=9, history=3)
    state = E.reset(1, cfg)
    prev = state
    for a in (0, 1, 2):
        prev = state
        state = E.step(state, jnp.array([a], jnp.int32), cfg)
    assert np.array_equal(np.asarray(state.hist[0, 0]), np.asarray(state.board[0]))
    assert np.array_equal(np.asarray(state.hist[0, 1]), np.asarray(prev.board[0]))
    # encode's most-recent own plane matches (board == to_play)
    cfg2 = TrainConfig.from_dict({"env": {"board_size": 9, "history": 3}})
    from bicgo.features import encode, make_pad_offsets

    pad = jnp.zeros((1,), jnp.int32)
    obs = encode(state, pad, pad, cfg2)
    assert np.array_equal(np.asarray(obs[0, :9, :9, 0] > 0),
                          np.asarray(state.board[0] == state.to_play[0]))


def test_reward_matches_outcome():
    cfg = EnvConfig(board_size=9, history=1, komi=7.5)
    st = E.reset(1, cfg)
    st = st._replace(passes=jnp.array([1], jnp.int32),
                     to_play=jnp.array([E.WHITE], jnp.int8))
    prev = st
    nxt = E.step(st, jnp.array([cfg.board_size ** 2], jnp.int32), cfg, score_now=False)
    just = nxt.done & ~prev.done
    assert bool(just[0])
    nxt = E.finalize_scores(nxt, just, cfg.komi)
    # White made the terminal pass and wins the empty board -> reward +1
    assert int(nxt.winner[0]) == E.WHITE
    assert float(E.terminal_reward(nxt, prev)[0]) == 1.0


def test_value_eval():
    from bicgo.arena import make_value_eval

    cfg = _cfg()
    key = jax.random.PRNGKey(6)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    value_eval = make_value_eval(model, cfg)
    out = jax.device_get(value_eval(ts.params, key))
    for k in ("ev_mean", "ev_actual", "value_mse", "value_acc", "ev_explained"):
        assert np.isfinite(float(out[k]))
    assert float(out["ev_positions"]) > 0
    assert 0.0 <= float(out["value_acc"]) <= 1.0


def test_sgf_roundtrip():
    from bicgo.sgf import index_to_vertex, moves_to_sgf, sgf_to_game, vertex_to_index

    moves = [0, 18, 40, 361, 180]
    sgf = moves_to_sgf(moves, 19, 7.5)
    size, komi, parsed = sgf_to_game(sgf)
    assert size == 19 and abs(komi - 7.5) < 1e-9
    assert [i for _, i in parsed] == moves
    for idx in (0, 3, 18, 360, 180):
        assert vertex_to_index(index_to_vertex(idx, 19), 19) == idx
    assert vertex_to_index("pass", 19) == 361
    assert vertex_to_index("bad", 19) is None


def test_entropy_schedule():
    from bicgo.train import entropy_coef_at

    cfg = _cfg()
    start = cfg.ppo.entropy_coef_start
    end = cfg.ppo.entropy_coef_end if cfg.ppo.entropy_coef_end is not None \
        else cfg.ppo.entropy_coef
    assert abs(entropy_coef_at(0, cfg) - start) < 1e-9
    assert abs(entropy_coef_at(cfg.iterations, cfg) - end) < 1e-9


def test_resume_roundtrip(tmp_path):
    cfg = _cfg()
    key = jax.random.PRNGKey(5)
    model = build_model(cfg)
    ts = create_train_state(model, cfg, key)
    rollout = make_rollout(model, cfg)
    update = make_update(model, cfg)
    state = E.reset(cfg.num_envs, cfg)
    tr, state, lv, _ = rollout(ts.params, state, key)
    from bicgo.ppo import compute_gae

    adv, tgt = compute_gae(tr.value, tr.reward, tr.done, lv, 1.0, 0.9)
    ts, _ = update(ts, tr, adv, tgt, key, jnp.float32(0.05))

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, ts, cfg, 7)
    ts2, it = load_train_state(ckpt, model, cfg, key)
    assert it == 7
    for a, b in zip(jax.tree_util.tree_leaves(ts.params),
                    jax.tree_util.tree_leaves(ts2.params)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(ts.step), np.asarray(ts2.step))
