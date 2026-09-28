"""Typed configuration for bicgo.

Everything is a plain dataclass so it can be constructed in Python, dumped to
JSON and reloaded, and used as a static argument for jit/vmap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvConfig:
    board_size: int = 19          # 9 / 13 / 19
    history: int = 8              # number of past board positions fed to the net
    komi: float = 7.5             # Chinese area komi ("黑贴3又3/4子")
    # Long hard cap on game length; reaching it ends the game as a DRAW. The
    # rollout budget never truncates a game -- unfinished games are value
    # bootstrapped and continue in the next rollout.
    max_moves: int = 4 * 19 * 19
    two_pass_end: bool = True     # classic rule: two consecutive passes end the game
    allow_suicide: bool = False   # HullQin: suicide is illegal


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 128           # feature width, constant across all stages
    res_blocks_pre: int = 4       # ResNet blocks before the ViT stage
    res_blocks_post: int = 4      # ResNet blocks after the ViT stage
    vit_depth: int = 4            # transformer blocks
    vit_heads: int = 8
    vit_mlp_ratio: int = 4
    value_hidden: int = 256       # width of the value MLP
    group_norm_groups: int = 8
    dropout: float = 0.0
    remat: bool = False           # gradient rematerialization (saves memory, costs time)
    attn_impl: str = "xla"        # "cudnn" uses flash attention (GPU) for a large speedup


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01     # final entropy coefficient
    entropy_coef_start: float | None = 0.08  # high early exploration; None -> entropy_coef
    entropy_coef_end: float | None = None    # None -> entropy_coef
    entropy_schedule_iters: int | None = None  # None -> TrainConfig.iterations
    kl_target: float = 0.03        # stop the remaining epochs once KL exceeds this
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_warmup_iters: int = 200
    lr_total_iters: int | None = None  # None -> TrainConfig.iterations
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch_size: int = 1024    # transitions per SGD minibatch (memory knob)
    normalize_advantage: bool = True
    seed: int = 41


@dataclass(frozen=True)
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    num_envs: int = 64            # parallel self-play games per iteration
    rollout_len: int = 256        # steps collected per iteration
    iterations: int = 1000
    log_every: int = 1
    eval_every: int = 50
    eval_games: int = 32
    eval_opponent: str = "random"  # random | self | pass
    eval_value: bool = True        # log value-head EV / calibration each eval
    value_eval_moves: int = 1000   # play value-eval games this long (to reach a real ending)
    ckpt_every: int = 50
    out_dir: str = "runs/bicgo"
    dtype: str = "bfloat16"       # float32 | bfloat16 (params in float32)
    randomize_pad: bool = True    # pad 19x19 -> 20x20 with random corner (augmentation)
    device: int = 0
    log_jsonl: bool = True        # append metrics/eval/ladder JSONL under out_dir
    comp_cache: bool = True       # JAX persistent compilation cache
    resume: str | None = None     # checkpoint directory to resume from

    # ---- Elo ladder ----
    ladder: bool = True
    ladder_games: int = 32        # games per opponent per evaluation
    ladder_opponents: int = 4     # snapshots to play against each evaluation
    ladder_max_moves: int = 361   # move cap for ladder games (area score tiebreak)
    ladder_k: float = 16.0        # Elo K-factor
    ladder_keep: int = 32         # maximum snapshots retained

    # ---- derived helpers -------------------------------------------------
    @property
    def pad(self) -> int:
        return 1  # 20x20 target for a 19x19 board

    @property
    def board_size(self) -> int:
        return self.env.board_size

    @property
    def padded_size(self) -> int:
        return self.env.board_size + self.pad

    @property
    def in_channels(self) -> int:
        return 2 * self.env.history + 3

    @property
    def num_actions(self) -> int:
        return self.env.board_size * self.env.board_size + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        return cls(
            env=_fill(EnvConfig, d.get("env", {})),
            model=_fill(ModelConfig, d.get("model", {})),
            ppo=_fill(PPOConfig, d.get("ppo", {})),
            **{k: v for k, v in d.items() if k in _top_fields()},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Backwards-friendly alias.
Config = TrainConfig


def _top_fields() -> set[str]:
    return {f.name for f in fields(TrainConfig) if f.name not in {"env", "model", "ppo"}}


def _fill(dc, d: dict[str, Any]):
    valid = {f.name for f in fields(dc)}
    return dc(**{k: v for k, v in d.items() if k in valid})
