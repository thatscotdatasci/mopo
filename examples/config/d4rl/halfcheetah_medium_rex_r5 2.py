from .halfcheetah_medium import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'break_train_rex': False,
    'rex': True,
    'policy_type': 'random_5',
    'repeat_dynamics_epochs': 1,
})


