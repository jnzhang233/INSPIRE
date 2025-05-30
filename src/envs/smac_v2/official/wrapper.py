from .distributions import get_distribution
from .starcraft2 import StarCraft2Env
from .starcraft2_hxt import StarCraft2Env as StarCraft2EnvMoveWithFov
from envs.multiagentenv import MultiAgentEnv


class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        change_fov_with_move = kwargs.pop("change_fov_with_move")
        self.env = StarCraft2EnvMoveWithFov(**kwargs) if change_fov_with_move else StarCraft2Env(**kwargs)
        assert (
                self.distribution_config.keys()
                == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        reset_config = {}
        for distribution in self.env_key_to_distribution_map.values():
            reset_config = {**reset_config, **distribution.generate()}

        return self.env.reset(reset_config)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()

    #__________________________________________________
    def get_indi_terminated(self):
        return self.env.get_indi_terminated()

    def get_ally_visibility_matrix(self):
        return self.env.get_ally_visibility_matrix()

    def get_unit_type_id_ICES(self):
        return self.env.get_unit_type_id_ICES()

    def get_state_ICES(self):
        return self.env.get_state_ICES()

    def get_obs_ICES(self):
        return self.env.get_obs_ICES()


