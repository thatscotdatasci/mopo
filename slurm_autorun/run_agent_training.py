import os
import json
import sys
import time
from copy import deepcopy
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from dogo.results import get_experiment_details

TIME_FORMAT = '%Y%m%d%H%M%S'
SLRUM_AUTORUN_DIR = os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/slurm_autorun')
SLURM_TRAIN_TEMPLATE_PATH = 'train.slurm.peta4-icelake.j2'

# Map rollout lengths to rollout batch sizes
# The following calculation is performed in the `_reallocate_model_pool` function of mopo.py:
#
# rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
# model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
# new_pool_size = self._model_retain_epochs * model_steps_per_epoch
#
# Thus, to keep the total pool size constant across experiments, `rollout_batch_size` needs
# to decrease as `_rollout_length` increases.
ROLLOUT_BATCH_SIZES = {
    5: 50000,
    10: 25000,
    20: 12500,
    50: 5000,
    100: 2500,
}


class MopoAgentExp:
    def __init__(
        self,
        config,
        mopo_penalty_coeff,
        rollout_length,
        dataset,
        output_dir,
        exp_id,
        exp_name=None,
        seed=None,
        bnn_retrain_epochs=0,
        rollout_batch_size=None
    ) -> None:
        """ This object is used to define a MOPO experiment.
        """
        # Parameters that must always be specified by the user
        self.config = config
        self.mopo_penalty_coeff = mopo_penalty_coeff
        self.rollout_length = rollout_length
        self.dataset = dataset
        self.exp_id = exp_id

        # Parameters with defaults, or values that are taken from the dynamics experiment
        self.exp_name = exp_name
        self.exp_name = exp_name + '_' + str(seed)
        self.seed = seed or dynamics_exp.seed
        self.bnn_retrain_epochs = bnn_retrain_epochs
        self.rollout_batch_size = rollout_batch_size or ROLLOUT_BATCH_SIZES[self.rollout_length]

        # Other arguments
        self.output_dir = output_dir

    _jinja_template = None
    @property
    def jinja_template(self):
        if self._jinja_template is None:
            template_loader = FileSystemLoader(searchpath=SLRUM_AUTORUN_DIR)
            env = Environment(loader=template_loader)
            self._jinja_template = env.get_template(SLURM_TRAIN_TEMPLATE_PATH)
        return self._jinja_template

    @property
    def exp_record(self):
        return {
            "config": self.config,
            "exp_name": self.exp_name,
            "seed": self.seed,
            "bnn_retrain_epochs": self.bnn_retrain_epochs,
            "mopo_penalty_coeff": self.mopo_penalty_coeff,
            "rollout_length": self.rollout_length,
            "rollout_batch_size": self.rollout_batch_size,
            "dataset": self.dataset,
        }

    @property
    def slurm_tmp_filename(self):
        t_stamp = datetime.utcfromtimestamp(int(time.time())).strftime(TIME_FORMAT)
        return os.path.join(SLRUM_AUTORUN_DIR, 'output', self.output_dir, f'train.slurm.peta4-icelake.{t_stamp}')
        
    def run_experiment(self):
        # Create the SLURM submission script, with the appropriate experiment parameters
        slrum_tmp_file = self.slurm_tmp_filename
        with open(slrum_tmp_file, 'w') as f:
            f.write(self.jinja_template.render(self.exp_record))

        # Run the SLURM sbatch command
        stream = os.popen(f'sbatch {slrum_tmp_file}')
        output = stream.read()

        # Verify that the message returned by the sbatch command is of the expected format
        if not output.startswith('Submitted batch job '):
            raise RuntimeError(f'Did not trigger job successfully!\n{self.exp_record}')

        # Extract the job ID returned by the sbatch command
        job_id = output.split()[-1]
        with open(slrum_tmp_file+'.' + str(job_id), 'w') as f:
            f.write('')

        print(job_id)
    
        return job_id

def run_experiment_set(params_filepath):
    # Create an output directory
    output_dir = params_filepath.split('/')[-1].replace('.json', '')
    os.makedirs(os.path.join(SLRUM_AUTORUN_DIR, 'output', output_dir))

    # Load the collection of experiments to be triggered, as defined in the parameters file
    with open(params_filepath, 'r') as f:
        exp_coll = json.load(f)
    
    failure = False
    exps_triggered = []
    exp_coll_triggered = []
    for exp_params in exp_coll:
        try:
            # Start the experiment running on the HPC
            job_id = MopoAgentExp(**exp_params, output_dir=output_dir).run_experiment()
        except RuntimeError:
            # Catch any failures
            failure = True
            break
        else:
            # Record that the experiment was triggered
            exps_triggered.append(job_id)

            # Take a copy of the experiment's configuration
            # Append the job ID returned by the HPC
            exp_triggered = deepcopy(exp_params)
            exp_triggered.update({'slurm_job_id': job_id})
            exp_coll_triggered.append(exp_triggered)

            # Pause between job submissions to avoid overloading the HPC CLI
            time.sleep(1)

    # Save the updated experiment config (including the HPC job ID) to the output directory
    with open(os.path.join(SLRUM_AUTORUN_DIR, 'output', output_dir, params_filepath.split('/')[-1]), 'w') as f:
        json.dump(exp_coll_triggered, f, indent=4)

    # Print the IDs of the experiments that have been triggered, in a format that will allow them to all be
    # cancelled in one go using the scancel command
    print(' '.join(exps_triggered))

    if failure:
        # Note that we do not attempt to automatically cancel jobs that have already been triggered
        raise RuntimeError('Did not trigger all jobs successfully!')

if __name__ == '__main__':
    # params_filepath = sys.argv[1]

    params_filepath = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/slurm_autorun/exp_params/exp_params.default.json")

    run_experiment_set(params_filepath)
