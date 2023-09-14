## adapted from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py

import os
import math
import pickle
from collections import OrderedDict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from mopo.models.constructor import construct_model, format_samples_for_training
from mopo.models.fake_env import FakeEnv
from mopo.utils.writer import Writer
from mopo.utils.visualization import visualize_policy
from mopo.utils.logging import Progress, Wandb
import mopo.utils.filesystem as filesystem
import mopo.off_policy.loader as loader


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class MOPO(RLAlgorithm):
    """Model-based Offline Policy Optimization (MOPO)

    References
    ----------
        Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma. 
        MOPO: Model-based Offline Policy Optimization. 
        arXiv preprint arXiv:2005.13239. 2020.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            bnn_lr=0.001,
            improvement_threshold=0.0001,
            break_train_rex=False,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            obs_indices=None,

            deterministic=False,
            rollout_random=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            # rollout_schedule=[20,100,1,1],
            rollout_length=1,
            hidden_dim=200,
            max_model_t=None,
            model_type='mlp',
            separate_mean_var=False,
            identity_terminal=0,

            pool_load_path='',
            pool_load_max_size=0,
            model_name=None,
            model_load_dir=None,
            penalty_coeff=0.,
            penalty_learned_var=False,

            # Project parameters
            rex=False,
            rex_beta=10.0,
            rex_multiply=False,
            rex_type='var',
            policy_type='default',
            holdout_policy=None,
            train_bnn_only=False,
            repeat_dynamics_epochs=1,
            lr_decay=1.0,
            bnn_batch_size=256,
            bnn_retrain_epochs=1,

            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            rex (`bool`): If True, we add the V-REx penalty to the loss during
                dynamics training.
            rex_beta (`float`): The penalty value to use in V-REx.
            rex_multiply (`bool`): If True, multiply variance by beta, else
                divide sum of losses by beta.
            holdout_policy (`float` or `None`): The policy to holdout during
                training and use for evaluation. If not specified, will select
                a subset of data from across policies randomly.
            train_bnn_only ('bool'): If True, only the BNN will be trained.
            repeat_dynamics_epochs ('int'): Number of epochs of dynamics model
                training to repeat again after convergence condition is met.
            lr_decay ('float'): (optional) Multiply the core loss by this number
                before returning. Applies in REx training loop.
            bnn_retrain_epochs ('int'): (optional) Number of epochs to retrain
                loaded BNN model for.
        """

        super(MOPO, self).__init__(**kwargs)

        self.obs_indices = obs_indices
        self._log_dir = os.getcwd()
        print('self._log_dir', self._log_dir)
        self._writer = Writer(self._log_dir)
        if not train_bnn_only:
            self.wparams = {**dict(training_environmen=training_environment,
                            evaluation_environment=evaluation_environment,
                            policy=policy, Qs=Qs, pool=pool, static_fns=static_fns, plotter=plotter,
                            tf_summaries=tf_summaries,
                            lr=lr,
                            reward_scale=reward_scale,
                            target_entropy=target_entropy,
                            discount=discount,
                            tau=tau,
                            target_update_interval=target_update_interval,
                            action_prior=action_prior,
                            reparameterize=reparameterize,
                            store_extra_policy_info=store_extra_policy_info,
                            deterministic=deterministic,
                            rollout_random=rollout_random,
                            model_train_freq=model_train_freq,
                            num_networks=num_networks,
                            num_elites=num_elites,
                            model_retain_epochs=model_retain_epochs,
                            rollout_batch_size=rollout_batch_size,
                            real_ratio=real_ratio,
                            rollout_length=rollout_length,
                            hidden_dim=hidden_dim,
                            max_model_t=max_model_t,
                            model_type=model_type,
                            separate_mean_var=separate_mean_var,
                            identity_terminal=identity_terminal,
                            pool_load_path=pool_load_path,
                            pool_load_max_size=pool_load_max_size,
                            model_name=model_name,
                            model_load_dir=model_load_dir,
                            penalty_coeff=penalty_coeff,
                            penalty_learned_var=penalty_learned_var,
                            rex=rex,
                            rex_beta=rex_beta,
                            rex_multiply=rex_multiply,
                            rex_type=rex_type,
                            policy_type=policy_type,
                            holdout_policy=holdout_policy,
                            train_bnn_only=train_bnn_only,
                            repeat_dynamics_epochs=repeat_dynamics_epochs,
                            lr_decay=lr_decay,
                            bnn_batch_size=bnn_batch_size,
                            bnn_retrain_epochs=bnn_retrain_epochs),
                            **kwargs}
            print('wandb self._log_dir', self._log_dir)
            self.domain = self._log_dir.split('/')[-3]
            self.exp_seed = self._log_dir.split('/')[-1].split('_')[0]
            self.exp_name = self._log_dir.split('/')[-2]
            # print('creating wandb logger policy!!!')
            # self.wlogger = Wandb(self.wparams, group_name=self.exp_name, name=self.exp_seed, project='Diversity_Policy')

        obs_dim = np.prod(training_environment.active_observation_shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._model_type = model_type
        self._identity_terminal = identity_terminal
        print('model_load_dir', model_load_dir)
        print('deterministic', deterministic)
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites,
                                      model_type=model_type, separate_mean_var=separate_mean_var,
                                      name=model_name, load_dir=model_load_dir, deterministic=deterministic,
                                      rex=rex, rex_beta=rex_beta, rex_multiply=rex_multiply, 
                                      lr_decay=lr_decay, log_dir=self._log_dir,
                                      train_bnn_only=train_bnn_only, rex_type=rex_type,
                                      policy_type=policy_type, bnn_lr=bnn_lr, improvement_threshold=improvement_threshold,
                                      break_train_rex=break_train_rex,
                                      wlogger=None, obs_indices=obs_indices)
        self._static_fns = static_fns
        self.fake_env = FakeEnv(self._model, self._static_fns, penalty_coeff=penalty_coeff,
                                penalty_learned_var=penalty_learned_var)

        self._rollout_schedule = [20, 100, rollout_length, rollout_length]
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._rollout_random = rollout_random
        self._real_ratio = real_ratio

        self._holdout_policy = holdout_policy
        self._repeat_dynamics_epochs = repeat_dynamics_epochs

        self._train_bnn_only = train_bnn_only
        self._bnn_batch_size = bnn_batch_size
        self._bnn_retrain_epochs = bnn_retrain_epochs

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        print('pool', pool)
        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MOPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

        #### load replay pool data
        self._pool_load_path = pool_load_path
        self._pool_load_max_size = pool_load_max_size

        loader.restore_pool(self._pool, self._pool_load_path, self._pool_load_max_size,
                            save_path=self._log_dir, policy_type=policy_type)
        self._init_pool_size = self._pool.size
        print('[ MOPO ] Starting with pool size: {}'.format(self._init_pool_size))
        ####

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

    def _train(self):
        
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        print('start training')
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        model_metrics = {}

        if not self._training_started:
            self._init_training()

        self.sampler.initialize(training_environment, policy, pool)
        print(' self.sampler',  self.sampler)
        print('self.sampler._max_path_length', self.sampler._max_path_length)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        #### model training
        print('[ MOPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
        print('[ MOPO ] Training model at epoch {} | freq {} | timestep {} (total: {})'.format(
            self._epoch, self._model_train_freq, self._timestep, self._total_timestep)
        )

        # The original MOPO code would run 1 epoch of training against a loaded model
        # Changes made to the code mean that we can now specify `self._bnn_retrain_epochs=0`
        print('save_path', os.path.join(self._log_dir, 'models'))
        max_epochs = self._bnn_retrain_epochs if self._model.model_loaded else None
        print('train model')
        model_train_metrics = self._train_model(
            batch_size=self._bnn_batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2,
            max_t=self._max_model_t,
            holdout_policy=self._holdout_policy,
            repeat_dynamics_epochs=self._repeat_dynamics_epochs
        )

        model_metrics.update(model_train_metrics)
        self._log_model()
        print('finished training model')
        gt.stamp('epoch_train_model')

        # If we are only learning a dynamics model, tell Ray to stop training at this point
        # No policy training will take place
        if self._train_bnn_only:
            yield {'done': True, **{}}
        ####

        print('creating wandb logger policy!!!')
        self.wlogger = Wandb(self.wparams, group_name=self.exp_name, name=self.exp_seed, project='Diversity_Policy')

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):

            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            # _n_train_repeat is 1 by default
            self._training_progress = Progress(self._epoch_length * self._n_train_repeat)
            start_samples = self.sampler._total_samples
            for timestep in count():
                self._timestep = timestep

                if (timestep >= self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                ## model rollouts
                # _real_ratio is 0.05 by default
                if timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                    self._training_progress.pause()
                    self._set_rollout_length()
                    print('_reallocate_model_pool')
                    self._reallocate_model_pool()
                    print('model rollout')
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                                                deterministic=self._deterministic)
                    print('model rollout finished')
                    model_metrics.update(model_rollout_metrics)
                    time_step_global = self._epoch_length * self._epoch + timestep

                    gt.stamp('epoch_rollout_model')
                    self._training_progress.resume()

                    # Save the model pool every 100 epochs, including the very first
                    if self._epoch % 100 == 0:
                        self._log_model_pool()

                ## train actor and critic
                if self.ready_to_train:
                    self._do_training_repeats(timestep=timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))

            print('evaluating policy')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment, obs_indices=self.obs_indices)
            print('evaluated policy finished')

            gt.stamp('evaluation_paths')

            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            # Evaluate the policy against the learned environment model
            # model_metrics.update(self._eval_model())

            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            if self._epoch % 10 == 0:
                diagnostics.update(OrderedDict((
                    *(
                        (f'evaluation/{key}', evaluation_metrics[key])
                        for key in sorted(evaluation_metrics.keys())
                    ),
                    *(
                        (f'times/{key}', time_diagnostics[key][-1])
                        for key in sorted(time_diagnostics.keys())
                    ),
                    *(
                        (f'sampler/{key}', sampler_diagnostics[key])
                        for key in sorted(sampler_diagnostics.keys())
                    ),
                    *(
                        (f'model/{key}', model_metrics[key])
                        for key in sorted(model_metrics.keys())
                    ),
                    *(('rollout_model/' + key, value) for key, value in model_rollout_metrics.items()),
                    ('epoch', self._epoch),
                    ('timestep', self._timestep),
                    ('timesteps_total', self._total_timestep),
                    ('train-steps', self._num_train_steps),
                    ('time_step_global', time_step_global)
                )))

                # print('logging diagnostics!')
                # print('diagnostics', diagnostics)
                self.wlogger.wandb.log(diagnostics, step=self._total_timestep)

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            ## ensure we did not collect any more data
            assert self._pool.size == self._init_pool_size

            yield diagnostics

        # Save the final model pool
        self._log_model_pool()

        self.sampler.terminate()

        self._training_after_hook()

        self._training_progress.close()

        yield {'done': True, **diagnostics}

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    def _log_model(self):
        print('MODEL: {}'.format(self._model_type))
        if self._model_type == 'identity':
            print('[ MOPO ] Identity model, skipping save')
        # Disabled the original code below - still want to save the loaded model
        # as it may have recieved at least one additional epoch of training (depending
        # on the value of `bnn_retrain_epochs`)
        # elif self._model.model_loaded:
        #     print('[ MOPO ] Loaded model, skipping save')
        else:
            save_path = os.path.join(self._log_dir, 'models')
            filesystem.mkdir(save_path)
            print('[ MOPO ] Saving model to: {}'.format(save_path))
            self._model.save(save_path, self._total_timestep)

    def _log_model_pool(self):
        # This is quite data intensive, and so is disabled by default.
        # Turning it on/off should really be implemented as a command line argument for convenience.
        return
        # # Save 100k random records from the model pool
        # save_path = os.path.join(self._log_dir, 'models')
        # filesystem.mkdir(save_path)
        # full_path = os.path.join(save_path, 'model_pool_{}.npy'.format(self._total_timestep))
        # pool_samples = self._model_pool.random_batch(100000)
        # pool_arr = np.hstack([
        #     pool_samples['observations'],
        #     pool_samples['actions'],
        #     pool_samples['next_observations'],
        #     pool_samples['rewards'],
        #     pool_samples['terminals'],
        #     pool_samples['policies'],
        #     pool_samples['penalties'],
        # ])
        # np.save(full_path, pool_arr)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        # For standard HalfCheetah _epoch_length is 1000 and model_train_freq is 1000
        # Thus, we create one batch of rollouts per epoch
        #
        # By default, _rollout_length is 5, _rollout_batch_size is 50,0000 and _model_retain_epochs is 5
        # Thus, we get a pool size of 1.25e+06
        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ MOPO ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
        
        elif self._model_pool._max_size != new_pool_size:
            print('[ MOPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples(self.obs_indices)
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        if self._model_type == 'identity':
            print('[ MOPO ] Identity model, skipping model')
            model_metrics = {}
        else:
            env_samples = self._pool.return_all_samples(self.obs_indices)
            for field_name in env_samples:
                if 'observations' in field_name:
                    print('_train_model field_name', field_name, env_samples[field_name].shape)
                    env_samples[field_name][:, self.obs_indices] = 0
            train_inputs, train_outputs, train_policies = format_samples_for_training(env_samples)
            model_metrics = self._model.train(train_inputs, train_outputs, train_policies, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {} | Type: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size, self._model_type
        ))
        batch = self.sampler.random_batch(rollout_batch_size, obs_indices=self.obs_indices)
        obs = batch['observations']

        # obs[:, self.obs_indices] = 0

        steps_added = []
        unpenalised_rewards = []
        penalised_rewards = []
        penalties = []
        for i in range(self._rollout_length):
            obs_act = copy.deepcopy(obs)
            obs_act[:, self.obs_indices] = 0
            if not self._rollout_random:
                act = self._policy.actions_np(obs_act)
            else:
                act_ = self._policy.actions_np(obs_act)
                act = np.random.uniform(low=-1, high=1, size=act_.shape)

            if self._model_type == 'identity':
                next_obs = obs
                rew = np.zeros((len(obs), 1))
                term = (np.ones((len(obs), 1)) * self._identity_terminal).astype(np.bool)
                info = {}
            else:
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
            steps_added.append(len(obs))
            unpenalised_rewards.append(info['unpenalized_rewards'].flatten())
            penalised_rewards.append(info['penalized_rewards'].flatten())

            pen = info['penalty'] if info['penalty'] is not None else np.zeros_like(rew)
            penalties.append(pen.flatten())

            # Adding a policy identifier to the rollouts - this will not be used during SAC training
            pol = np.zeros((len(obs), 1))

            # print('add_samples next_obs', next_obs.shape)
            # next_obs[:, self.obs_indices] = 0

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term, 'policies': pol, 'penalties': pen}
            # print('add_samples samples observations shape', samples['observations'].shape)
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        # Horizontally stack all of the reward vectors - necessary when dealing with environments that have variable episode lengths.
        # The `np.mean` etc. methods attempt to perform a stacking, which fails - do the job for them.
        unpenalised_rewards = np.hstack(unpenalised_rewards)
        penalised_rewards = np.hstack(penalised_rewards)
        penalties = np.hstack(penalties)

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {
            'mean_rollout_length': mean_rollout_length,
            'mean_unpenalized_rewards': np.mean(unpenalised_rewards),
            'std_unpenalized_rewards': np.std(unpenalised_rewards),
            'min_unpenalized_rewards': np.min(unpenalised_rewards),
            'max_unpenalized_rewards': np.max(unpenalised_rewards),
            'mean_penalized_rewards': np.mean(penalised_rewards),
            'std_penalized_rewards': np.std(penalised_rewards),
            'min_penalized_rewards': np.min(penalised_rewards),
            'max_penalized_rewards': np.max(penalised_rewards),
            'mean_penalty': np.mean(penalties),
            'std_penalty': np.std(penalties),
            'min_penalty': np.min(penalties),
            'max_penalty': np.max(penalties),
        }
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats

    def _eval_model(self):
        # Sample starting locations from the training data
        # batch = self.sampler.random_batch(self._eval_n_episodes)
        # obs = batch['observations']
        
        # Sample starting locations from the real environment
        obs = np.vstack(self._evaluation_environment.reset()['observations'] for _ in range(self._eval_n_episodes))
        print('_eval_model obs shape', obs.shape)
        obs[:, self.obs_indices] = 0

        # Episodes can finish at different times - keep track of those that are still running
        nonterm_mask = np.ones(self._eval_n_episodes).astype(bool)
        unpenalised_rewards = []

        # 1000 is the max episode length for MuJoCo environments
        for i in range(1000):
            # Below is copied from `_rollout_model`, which generates data for SAC training
            if not self._rollout_random:
                act = self._policy.actions_np(obs)
            else:
                act_ = self._policy.actions_np(obs)
                act = np.random.uniform(low=-1, high=1, size=act_.shape)

            if self._model_type == 'identity':
                next_obs = obs
                rew = np.zeros((len(obs), 1))
                term = (np.ones((len(obs), 1)) * self._identity_terminal).astype(np.bool)
                info = {}
            else:
                next_obs, rew, term, info = self.fake_env.step(obs, act)

            next_obs[:, self.obs_indices] = 0
            # Determine those episodes that did not terminate in the current step
            step_nonterm_mask = ~term.squeeze(-1)

            # Update nonterm_mask to reflect the episodes that are still running
            # The purpose of nonterm_mask[nonterm_mask] is to update only those episodes that were running
            # at the start of the current step. Remember that nonterm_mask is a boolean array, and so can 
            # be used for slicing in this way.
            nonterm_mask[nonterm_mask] = step_nonterm_mask

            # Early exit if there are no episodes still running.
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            # Any episodes that have finished get a `np.nan` reward value in the current step
            # The remainder get the appropriate reward value
            step_unpen_rewards = np.ones(self._eval_n_episodes) * np.nan
            step_unpen_rewards[nonterm_mask] = info['unpenalized_rewards'].flatten()[step_nonterm_mask]
            unpenalised_rewards.append(step_unpen_rewards)

            # For increased efficieny, we only keep running with those episodes that have not already terminated
            obs = next_obs[step_nonterm_mask]

        unpenalised_rewards = np.vstack(unpenalised_rewards)
        unpenalised_returns = np.nansum(unpenalised_rewards, axis=0)

        # `eval_return_count` will include only those episodes that did not fail immediately
        # It is not anticipated that any episodes should fail immediately in the vast majority of cases
        rollout_stats = {
            'eval_return_mean': np.nanmean(unpenalised_returns),
            'eval_return_std': np.nanstd(unpenalised_returns),
            'eval_return_count': np.sum(~np.isnan(unpenalised_returns)),
            'eval_mean_length': np.mean(np.sum(~np.isnan(unpenalised_rewards), axis=0))
        }

        return rollout_stats

    def _visualize_model(self, env, timestep):
        ## save env state
        state = env.unwrapped.state_vector()
        qpos_dim = len(env.unwrapped.sim.data.qpos)
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]

        print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
        visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
        print('[ Visualization ] Done')
        ## set env state
        env.unwrapped.set_state(qpos, qvel)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        print('real self._pool', self._pool)
        env_batch = self._pool.random_batch(env_batch_size, obs_indices=self.obs_indices)

        if model_batch_size > 0:
            print('model self._model_pool', self._model_pool)
            model_batch = self._model_pool.random_batch(model_batch_size, obs_indices=self.obs_indices)

            # keys = env_batch.keys()
            keys = set(env_batch.keys()) & set(model_batch.keys())
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch

        for field_name in batch.keys():
            if 'observation' in field_name:
                print('_training_batch field_name', field_name)
                batch[field_name][:, self.obs_indices] = 0

        return batch

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        self._training_progress.update()
        self._training_progress.set_description()

        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q-min': np.min(Q_values),
            'Q-25': np.percentile(Q_values, 25),
            'Q-50': np.percentile(Q_values, 50),
            'Q-75': np.percentile(Q_values, 75),
            'Q-95': np.percentile(Q_values, 95),
            'Q-max': np.max(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
