"""Shape / invariance checks for the network and GAE correctness."""
import numpy as np
import jax
import jax.numpy as jnp

from bicgo.config import TrainConfig, EnvConfig
from bicgo.features import encode, random_pad_offsets, gather_board_logits
from bicgo.model import build_model, forward
from bicgo.ppo import compute_gae
from bicgo import go_env as E


def test_feature_shapes_and_pad():
    cfg = TrainConfig.from_dict({"env": {"board_size": 19, "history": 4}})
    state = E.reset(3, cfg)
    key = jax.random.PRNGKey(0)
    pad_r, pad_c = random_pad_offsets(key, 3, cfg.pad)
    obs = encode(state, pad_r, pad_c, cfg)
    assert obs.shape == (3, 20, 20, cfg.in_channels)
    valid = np.asarray(obs[..., -1])
    for b in range(3):
        assert valid[b].sum() == 19 * 19
        r0, c0 = int(pad_r[b]), int(pad_c[b])
        assert valid[b, r0, c0] == 1
        assert valid[b, :r0, :].sum() == 0
        assert valid[b, r0 + 19 :, :].sum() == 0


def test_gather_board_logits():
    cfg = TrainConfig.from_dict({"env": {"board_size": 5}})
    s = cfg.padded_size
    padded = jnp.arange(s * s, dtype=jnp.float32).reshape(s, s)
    pad_r = jnp.array([0, 1], jnp.int32)
    pad_c = jnp.array([1, 0], jnp.int32)
    out = gather_board_logits(padded[None].repeat(2, 0), pad_r, pad_c, cfg)
    assert out.shape == (2, 25)
    assert np.array_equal(np.asarray(out[0]).reshape(5, 5), np.asarray(padded)[0:5, 1:6])
    assert np.array_equal(np.asarray(out[1]).reshape(5, 5), np.asarray(padded)[1:6, 0:5])


def test_model_forward_shapes():
    cfg = TrainConfig.from_dict(
        {"model": {"channels": 16, "res_blocks_pre": 1, "res_blocks_post": 1,
                   "vit_depth": 1, "vit_heads": 4, "value_hidden": 32},
         "env": {"board_size": 9, "history": 2}}
    )
    model = build_model(cfg)
    key = jax.random.PRNGKey(0)
    state = E.reset(2, cfg)
    pad_r, pad_c = random_pad_offsets(key, 2, cfg.pad)
    obs = encode(state, pad_r, pad_c, cfg)
    params = model.init(key, obs, obs[..., -1])["params"]
    logits, value = forward(model, params, obs, obs[..., -1], pad_r, pad_c, cfg)
    assert logits.shape == (2, cfg.num_actions)
    assert value.shape == (2,)
    assert np.all(np.abs(np.asarray(value)) <= 1.0)


def test_gae_matches_manual():
    gamma, lam = 1.0, 0.9
    T, B = 4, 2
    rng = np.random.default_rng(0)
    values = jnp.asarray(rng.normal(size=(T, B)), jnp.float32)
    rewards = jnp.zeros((T, B), jnp.float32)
    dones = jnp.zeros((T, B), jnp.float32)
    rewards = rewards.at[T - 1, 0].set(1.0).at[T - 1, 1].set(-1.0)
    dones = dones.at[T - 1].set(1.0)
    last_value = jnp.asarray(rng.normal(size=(B,)), jnp.float32)

    adv, tgt = compute_gae(values, rewards, dones, last_value, gamma, lam)
    adv = np.asarray(adv)
    tgt = np.asarray(tgt)

    v = np.asarray(values)
    r = np.asarray(rewards)
    d = np.asarray(dones)
    lv = np.asarray(last_value)
    delta = np.zeros((T, B))
    for t in range(T):
        v_next = lv if t == T - 1 else v[t + 1]
        nonterminal = 1.0 - d[t]
        delta[t] = r[t] + gamma * nonterminal * (-v_next) - v[t]
    manual = np.zeros((T, B))
    for t in range(T - 1, -1, -1):
        acc = delta[t]
        if t + 1 < T:
            nonterminal = 1.0 - d[t]
            acc += (-gamma * lam) * nonterminal * manual[t + 1]
        manual[t] = acc
    assert np.allclose(adv, manual, atol=1e-5)
    assert np.allclose(tgt, manual + v, atol=1e-5)


def test_gae_terminal_target():
    # one-move game, black to move and wins: target for the terminal step = +1
    values = jnp.array([[0.2]], jnp.float32)
    rewards = jnp.array([[1.0]], jnp.float32)
    dones = jnp.array([[1.0]], jnp.float32)
    last_value = jnp.array([0.7], jnp.float32)
    adv, tgt = compute_gae(values, rewards, dones, last_value, 1.0, 0.9)
    assert np.allclose(np.asarray(tgt), 1.0, atol=1e-6)
