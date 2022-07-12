"""Provides functions that are utilized by the command line interface.

In particular, the examples are exposed to the command line interface
(defined in `softlearning.scripts.console_scripts`) through the
`get_trainable_class`, `get_variant_spec`, and `get_parser` functions.
"""
import json

RESULTS_MAP_PATH = '/home/ajc348/rds/hpc-work/dogo_results/mopo/results_map.json'

def get_trainable_class(*args, **kwargs):
    from .main import ExperimentRunner
    return ExperimentRunner


# def get_variant_spec(command_line_args, *args, **kwargs):
#     from .variants import get_variant_spec
#     variant_spec = get_variant_spec(command_line_args, *args, **kwargs)
#     return variant_spec

def get_params_from_file(
    filepath, seed=None, exp_name=None, dataset=None, dynamics_model_exp=None, penalty_coeff=None,
    rollout_length=None, rollout_batch_size=None, params_name='params'
):
	import importlib
	from dotmap import DotMap
	module = importlib.import_module(filepath)
	params = getattr(module, params_name)

    ####################################################
    # Command Line Parameter Overrides
    # For the below parameters, if values have been spe-
    # cified on the command line then these will take 
    # precidence.
    ####################################################
	if exp_name is not None:
		params['exp_name'] = exp_name

	if seed is not None:
		params['seed'] = seed

	if dataset is not None:
		params['kwargs']['pool_load_path'] = f'/home/ajc348/rds/hpc-work/dogo_results/data/{dataset}.npy'

	if penalty_coeff is not None:
		params['kwargs']['penalty_coeff'] = penalty_coeff

	if rollout_length is not None:
		params['kwargs']['rollout_length'] = rollout_length

	if rollout_batch_size is not None:
		params['kwargs']['rollout_batch_size'] = rollout_batch_size

	params['kwargs']['dynamics_model_exp'] = dynamics_model_exp
	if dynamics_model_exp is not None:
		# Identify the experiment folder from the provided name
		with open(RESULTS_MAP_PATH, 'r') as f:
			results_map = json.load(f)
		load_exp_details = results_map[dynamics_model_exp]
		params['kwargs']['model_load_dir'] = f'/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/{load_exp_details["environment"]}/{load_exp_details["base_dir"]}/{load_exp_details["experiment_dir"]}/models'

	params = DotMap(params)
	return params

def get_variant_spec(command_line_args, *args, **kwargs):
    from .base import get_variant_spec
    import importlib
    params = get_params_from_file(
        filepath=command_line_args.config,
        seed=command_line_args.seed,
        exp_name=command_line_args.exp_name,
        dataset=command_line_args.dataset,
        dynamics_model_exp=command_line_args.dynamics_model_exp,
        penalty_coeff=command_line_args.penalty_coeff,
        rollout_length=command_line_args.rollout_length,
        rollout_batch_size=command_line_args.rollout_batch_size,
    )
    # import pdb	
    # pdb.set_trace()
    variant_spec = get_variant_spec(command_line_args, *args, params, **kwargs)
    return variant_spec

def get_parser():
    from examples.utils import get_parser
    parser = get_parser()
    return parser
