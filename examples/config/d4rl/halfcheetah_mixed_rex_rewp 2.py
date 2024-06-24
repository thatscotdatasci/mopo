from .halfcheetah_mixed import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'break_train_rex': False,
    'rex': True,
    'policy_type': 'reward_partioned',
    'repeat_dynamics_epochs': 1,
})

