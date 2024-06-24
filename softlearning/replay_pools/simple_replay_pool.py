from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete
import pdb

from .flexible_replay_pool import FlexibleReplayPool


def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space

        active_size = sum(
            np.prod(observation_space.spaces[key].shape)
            for key in list(observation_space.spaces.keys()))

        active_observation_shape = (active_size, )

        fields = {
            'actions': {
                'shape': self._action_space.shape,
                'dtype': 'float32'
            },
            'rewards': {
                'shape': (1, ),
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': (1, ),
                'dtype': 'bool'
            },
            'observations': {
                'shape': active_observation_shape,
                'dtype': 'float32'
            },
            'next_observations': {
                'shape': active_observation_shape,
                'dtype': 'float32'
            },
            'policies': {
                'shape': (1, ),
                'dtype': 'float32'
            },
            'penalties': {
                'shape': (1, ),
                'dtype': 'float32'
            },
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        return super(SimpleReplayPool, self).add_samples(samples)

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None,
                         obs_indices=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        # batch_ = {
        #     field_name: self.fields[field_name][indices]
        #     for field_name in self.field_names
        # }

        batch = {}

        for field_name in self.field_names:
            batch[field_name] = self.fields[field_name][indices]
            # if 'observation' in field_name and obs_indices is not None:
            #     batch[field_name][:, obs_indices] = 0
                # print('batch[field_name]', batch[field_name].shape)
                # print("batch[field_name][:, obs_indices]", batch[field_name][:, obs_indices].shape)
                # print("batch[field_name][:, obs_indices]", batch[field_name][:3, obs_indices])

        # for field_name in self.field_names:
        #     print(field_name, '(batch[field_name] == batch_[field_name]).all()',
        #           (batch[field_name] == batch_[field_name]).all())

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass
