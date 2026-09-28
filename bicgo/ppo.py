"""PPO objective and GAE for two-player self-play.

Only the terminal step carries a non-zero reward (``+1`` win / ``-1`` loss for
the player who moved). ``gamma = 1.0`` and ``lambda = 0.9`` by default.

The value head predicts the outcome from the perspective of the player to move,
so bootstrapping from state ``s_{t+1}`` (where the opponent is to move) flips
the sign. Episode boundaries (auto-reset after a terminal step) stop the GAE
recursion via the ``done`` mask.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .model import compute_dtype, forward, masked_log_probs


class Transition(NamedTuple):
    obs: jnp.ndarray          # (T, B, S, S, C)
    valid: jnp.ndarray        # (T, B, S, S)
    legal: jnp.ndarray        # (T, B, A)
    pad_r: jnp.ndarray        # (T, B)
    pad_c: jnp.ndarray        # (T, B)
    action: jnp.ndarray       # (T, B)
    log_prob: jnp.ndarray     # (T, B)
    value: jnp.ndarray        # (T, B)
    reward: jnp.ndarray       # (T, B)
    done: jnp.ndarray         # (T, B)


def compute_gae(values, rewards, dones, last_value, gamma: float, lam: float):
    """Return ``(advantages, value_targets)`` with shape ``(T, B)``.

    ``values``/``rewards``/``dones`` are ``(T, B)``; ``last_value`` is ``(B,)``
    and holds the canonical value of the state after the final action.
    """
    # The value is from the perspective of the player to move, and the players
    # alternate, so the effective per-step discount is ``-gamma``.  Hence the
    # GAE recursion subtracts the (gamma*lambda)-discounted next advantage:
    #   G_t - V_t = sum_l (-gamma)^l delta_{t+l}
    def step(carry, xs):
        adv_next, v_next = carry
        v_t, r_t, d_t = xs
        nonterminal = 1.0 - d_t
        bootstrap = nonterminal * (-v_next)          # opponent is to move next
        delta = r_t + gamma * bootstrap - v_t
        adv = delta - gamma * lam * nonterminal * adv_next
        return (adv, v_t), adv

    init = (jnp.zeros_like(last_value), last_value)
    (_, _), advs = lax.scan(
        step, init, (values[::-1], rewards[::-1], dones[::-1])
    )
    advs = advs[::-1]
    return advs, advs + values


def ppo_loss(model, params, batch: Transition, advantages, targets, cfg,
             entropy_coef, rng=None):
    """PPO clipped surrogate + value + entropy, averaged over the minibatch.

    ``entropy_coef`` is a traced scalar so the entropy bonus can be scheduled
    without recompiling.
    """
    obs = batch.obs.astype(compute_dtype(cfg.dtype))
    logits, value = forward(
        model, params, obs, batch.valid, batch.pad_r, batch.pad_c, cfg,
        deterministic=cfg.model.dropout <= 0.0,
    )
    logp_all = masked_log_probs(logits, batch.legal)
    logp = jnp.take_along_axis(logp_all, batch.action[:, None], axis=-1)[:, 0]

    ratio = jnp.exp(logp - batch.log_prob)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - cfg.ppo.clip_eps, 1.0 + cfg.ppo.clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    value_loss = jnp.mean((value - targets) ** 2)

    probs = jnp.exp(logp_all)
    entropy = -jnp.sum(jnp.where(batch.legal, probs * logp_all, 0.0), axis=-1)
    entropy_loss = -jnp.mean(entropy)

    loss = (
        policy_loss
        + cfg.ppo.value_coef * value_loss
        + entropy_coef * entropy_loss
    )
    approx_kl = jnp.mean(batch.log_prob - logp)
    clip_frac = jnp.mean(
        jnp.abs(ratio - 1.0) > cfg.ppo.clip_eps
    )
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": jnp.mean(entropy),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return loss, metrics
