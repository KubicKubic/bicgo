"""A lightweight Elo ladder for monitoring self-play strength.

The ladder keeps a bounded population of parameter snapshots with Elo ratings
(anchored at 0 for the initial network). Every evaluation the current model
plays a few games (colour-balanced, batched on the accelerator) against a
sample of past snapshots; ratings are updated with the standard Elo rule. This
gives a monotone-ish strength signal to watch during long training runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LadderEntry:
    iteration: int
    params: Any
    rating: float
    games: int = 0


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float):
    expected = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    return (
        rating_a + k * (score_a - expected),
        rating_b + k * (expected - score_a),
    )


class EloLadder:
    def __init__(self, cfg, play, seed_params, seed_iteration: int = 0):
        self.cfg = cfg
        self.play = play
        self.entries: list[LadderEntry] = [
            LadderEntry(seed_iteration, seed_params, 0.0, 0)
        ]
        self.rng = np.random.default_rng(cfg.ppo.seed)

    def _select(self) -> list[int]:
        m = len(self.entries)
        k = min(int(self.cfg.ladder_opponents), m)
        # always include the most recent snapshot, sample the rest uniformly
        chosen = {m - 1}
        if k > 1:
            pool = [i for i in range(m - 1)]
            if pool:
                extra = min(k - 1, len(pool))
                picks = self.rng.choice(pool, size=extra, replace=False)
                chosen.update(int(p) for p in picks)
        return sorted(chosen)

    def evaluate_and_update(self, params, iteration: int, key) -> tuple[float, dict]:
        idxs = self._select()
        rating = float(np.mean([self.entries[i].rating for i in idxs]))
        results: dict[int, float] = {}
        for i in idxs:
            entry = self.entries[i]
            score = self.play(
                params, entry.params, key, self.cfg.ladder_games, "policy"
            )
            rating, entry.rating = elo_update(
                rating, entry.rating, score, self.cfg.ladder_k
            )
            entry.games += self.cfg.ladder_games
            results[entry.iteration] = score
        self.entries.append(
            LadderEntry(iteration, params, rating, len(idxs) * self.cfg.ladder_games)
        )
        if len(self.entries) > self.cfg.ladder_keep:
            self.entries = self.entries[-self.cfg.ladder_keep :]
        return rating, results
