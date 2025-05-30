from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2 import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

try:
    ices_gfootball = True
    from .gfootball.ICES_FootballEnv import ICES_FootballEnv
except Exception as e:
    ices_gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V2 is not supported...")

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
else:
    print("GRF is not supported...")

if ices_gfootball:
    REGISTRY["ices_gfootball"] = partial(env_fn, env=ICES_FootballEnv)
else:
    print("ICES_GRF is not supported...")

print("Supported environments:", REGISTRY)
