"""bicgo: a pure-PPO JAX trainer for 19x19 Go.

Rules follow the HullQin (game.hullqin.cn) weiqi implementation:
  * board sizes 9/13/19 (default 19), black = 1, white = 2, empty = 0
  * black moves first
  * simple ko (not positional superko), suicide is illegal
  * pass is encoded as board index N*N (361 for 19x19)
  * Chinese area scoring with komi 7.5 ("黑贴3又3/4子")
"""

from .config import Config, EnvConfig, ModelConfig, PPOConfig, TrainConfig

__all__ = ["Config", "EnvConfig", "ModelConfig", "PPOConfig", "TrainConfig"]
__version__ = "0.1.0"
