import numpy as np
import tensorflow as tf
import pdb

from mopo.models.fc import FC
from mopo.models.bnn import BNN

def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7,
					num_elites=5, session=None, model_type='mlp', separate_mean_var=False,
					name=None, load_dir=None, deterministic=False, rex=False, rex_beta=10.0,
					rex_multiply=False, lr_decay=1.0, log_dir=None, train_bnn_only=False):
	if name is None:
		name = 'BNN'
	print('[ BNN ] Name {} | Observation dim {} | Action dim: {} | Hidden dim: {} | REx: {} | REx Beta: {}'.format(name, obs_dim, act_dim, hidden_dim, rex, rex_beta))
	params = {'name': name, 'num_networks': num_networks, 'num_elites': num_elites,
				'sess': session, 'separate_mean_var': separate_mean_var, 'deterministic': deterministic,
				'rex': rex, 'rex_beta': rex_beta, 'rex_multiply': rex_multiply, 'lr_decay': lr_decay,
				'log_dir': log_dir, 'train_bnn_only': train_bnn_only, 'rex_std': rex_std}

	if load_dir is not None:
		print('Specified load dir', load_dir)
		params['model_dir'] = load_dir
		params['load_model'] = True

	model = BNN(params)

	if not model.model_loaded:
		if model_type == 'identity':
			return
		elif model_type == 'linear':
			print('[ BNN ] Training linear model')
			model.add(FC(obs_dim+rew_dim, input_dim=obs_dim+act_dim, weight_decay=0.000025))
		elif model_type == 'mlp':
			print('[ BNN ] Training non-linear model | Obs: {} | Act: {} | Rew: {}'.format(obs_dim, act_dim, rew_dim))
			model.add(FC(hidden_dim, input_dim=obs_dim+act_dim, activation="swish", weight_decay=0.000025))
			model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
			model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
			model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
			model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))
			if separate_mean_var:
				model.add(FC(obs_dim+rew_dim, input_dim=hidden_dim, weight_decay=0.0001), var_layer=True)

	if load_dir is not None:
		model.model_loaded = True

	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	print('[ BNN ] Model: {}'.format(model))
	return model

def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	pol = samples['policies']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs, pol

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
