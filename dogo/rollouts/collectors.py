import numpy as np

class RolloutCollector:
    def __init__(self) -> None:
        self.obs = []
        self.acts = []
        self.next_obs = []
        self.rews = []

    def add_transition(self, obs, act, next_obs, rew):
        if type(obs) == dict:
            obs = obs['observations']
        if type(next_obs) == dict:
            next_obs = next_obs['observations']
        if type(rew) == np.array:
            rew = rew[0,0]

        self.obs.append(obs)
        self.acts.append(act)
        self.next_obs.append(next_obs)
        self.rews.append(rew)

    def return_transitions(self):
        return {
            'obs':      np.vstack(self.obs),
            'acts':     np.vstack(self.acts),
            'next_obs': np.vstack(self.next_obs),
            'rewards':  np.vstack(self.rews),
        }

class MopoRolloutCollector(RolloutCollector):
    def __init__(self) -> None:
        super().__init__()
        self.unpen_rews = []
        self.rew_pens = []
        self.ensemble_means_mean = []
        self.ensemble_means_std = []
        self.ensemble_vars_mean = []
        self.ensemble_vars_std = []
        self.ensemble_vars_max = []

    def add_transition(self, obs, act, next_obs, rew, info):
        super().add_transition(obs, act, next_obs, rew)
        self.unpen_rews.append(info['unpenalized_rewards'])
        self.rew_pens.append(info['penalty'])
        self.ensemble_means_mean.append(info['ensemble_means_mean'])
        self.ensemble_means_std.append(info['ensemble_means_std'])
        self.ensemble_vars_mean.append(info['ensemble_vars_mean'])
        self.ensemble_vars_std.append(info['ensemble_vars_std'])
        self.ensemble_vars_max.append(info['ensemble_vars_max'])

    def return_transitions(self):
        trans = super().return_transitions()
        trans.update({
            'unpen_rewards': np.vstack(self.unpen_rews),
            'reward_pens': np.vstack(self.rew_pens),
            'ensemble_means_mean': np.vstack(self.ensemble_means_mean),
            'ensemble_means_std': np.vstack(self.ensemble_means_std),
            'ensemble_vars_mean': np.vstack(self.ensemble_vars_mean),
            'ensemble_vars_std': np.vstack(self.ensemble_vars_std),
            'ensemble_vars_max': np.vstack(self.ensemble_vars_max),
        })
        return trans