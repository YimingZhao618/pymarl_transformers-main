from functools import partial
import sys
import os
import warnings

from .multiagentenv import MultiAgentEnv
try:
    from .mpe.mpe_wrapper import MPEWrapper
except ImportError as exc:
    warnings.warn(f"MPE is unavailable: {exc}")
    MPEWrapper = None
try:
    from .starcraft import StarCraft2Env
    include_sc2 = True
except:
    warnings.warn("Impossible to import SMAC, verify your installation.")
    include_sc2 = False

try:
    from .smac_v2 import StarCraft2Env2Wrapper
    include_smacv2 = True
except Exception as exc:
    warnings.warn(f"SMACv2 is unavailable: {exc}")
    include_smacv2 = False
from .matrix_game import OneStepMatrixGame

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
if include_sc2: # include starcraft only if correctly installed
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
if include_smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
if MPEWrapper is not None:
    REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
