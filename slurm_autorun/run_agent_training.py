import os
import json
import sys
import time
from copy import deepcopy
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from dogo.results import get_experiment_details

TIME_FORMAT = '%Y%m%d%H%M%S'
SLRUM_AUTORUN_DIR = '/home/ajc348/rds/hpc-work/mopo/slurm_autorun'
SLURM_TRAIN_TEMPLATE_PATH = 'train.slurm.peta4-icelake.j2'

ROLLOUT_BATCH_SIZES = {
    5: 50000,
    10: 25000,
    20: 12500,
    50: 5000,
}


class MopoAgentExp:
    def __init__(
        self,
        config,
        dynamics_model_exp,
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
        dynamics_exp = get_experiment_details(dynamics_model_exp)

        # Params always specified by user
        self.config = config
        self.dynamics_model_exp = dynamics_model_exp
        self.mopo_penalty_coeff = mopo_penalty_coeff
        self.rollout_length = rollout_length
        self.dataset = dataset
        self.exp_id = exp_id

        # Params with defaults/values taken from the dynamics experiment
        self.exp_name = exp_name or '_'.join(dynamics_exp.base_dir.split('_')[:-1])
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
            "dynamics_model_exp": self.dynamics_model_exp,
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
        slrum_tmp_file = self.slurm_tmp_filename
        with open(slrum_tmp_file, 'w') as f:
            f.write(self.jinja_template.render(self.exp_record))
        stream = os.popen(f'sbatch {slrum_tmp_file}')
        output = stream.read()
            
        if not output.startswith('Submitted batch job '):
            raise RuntimeError(f'Did not trigger job successfully!\n{self.exp_record}')

        job_id = output.split()[-1]
        with open(slrum_tmp_file+'.' + str(job_id), 'w') as f:
            f.write('')

        print(job_id)
    
        return job_id

def run_experiment_set(params_filepath):
    output_dir = params_filepath.split('/')[-1].replace('.json', '')

    os.makedirs(os.path.join(SLRUM_AUTORUN_DIR, 'output', output_dir))

    with open(params_filepath, 'r') as f:
        exp_coll = json.load(f)
    
    failure = False
    exps_triggered = []
    exp_coll_triggered = []
    for exp_params in exp_coll:
        try:
            job_id = MopoAgentExp(**exp_params, output_dir=output_dir).run_experiment()
        except RuntimeError:
            failure = True
            break
        else:
            exps_triggered.append(job_id)

            exp_triggered = deepcopy(exp_params)
            exp_triggered.update({'slurm_job_id': job_id})
            exp_coll_triggered.append(exp_triggered)

            time.sleep(1)

    with open(os.path.join(SLRUM_AUTORUN_DIR, 'output', output_dir, params_filepath.split('/')[-1]), 'w') as f:
        json.dump(exp_coll_triggered, f, indent=4)

    print(' '.join(exps_triggered))

    if failure:
        raise RuntimeError('Did not trigger all jobs successfully!')

if __name__ == '__main__':
    # params_filepath = sys.argv[1]

    params_filepath = "/home/ajc348/rds/hpc-work/mopo/slurm_autorun/exp_params/exp_params.250922_1.json"

    run_experiment_set(params_filepath)
