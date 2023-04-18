base_params = {
    'type': 'MOPO',
    'universe': 'gym',

    'log_dir': './ray_mopo/', # Specify where to write log files here

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,
        'policy_type': 'default', #random_5 random default

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'hidden_dim': 200,
        'bnn_lr': 0.001,  # 0.001
        'improvement_threshold': 0.001,
        'break_train_rex': False,
        # 'hidden_dim': 2048,
        # 'bnn_lr': 0.0001, #0.001
        # 'improvement_threshold': 0.00001,
        # 'break_train_rex': False,

        'model_train_freq': 1000,
        'model_retain_epochs': 5,
        'rollout_batch_size': 50e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -3,
        'max_model_t': None
    }
}