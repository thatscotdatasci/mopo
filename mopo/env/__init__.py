import sys

from .ant import AntEnv
from .humanoid import HumanoidEnv

# env_overwrite = {'Ant': AntEnv, 'Humanoid': HumanoidEnv}
# env_overwrite = {'Ant': AntEnv}
env_overwrite = {}

sys.modules[__name__] = env_overwrite