import os
import json
import sys
import time
from datetime import datetime
import itertools

from jinja2 import Environment, FileSystemLoader

from dogo.results import get_experiment_details

TIME_FORMAT = '%Y%m%d%H%M%S'
SLRUM_AUTORUN_DIR = '/home/ajc348/rds/hpc-work/mopo/slurm_autorun'
SLURM_TRAIN_TEMPLATE_PATH = 'train.slurm.peta4-icelake.j2'

ROLLOUT_BATCH_SIZES = {
    5: 50000,
    10: 25000,
    20: 12500,
}


class MopoExpSet:
    def __init__(
        self,
        config,
        dynamics_model_exps,
        mopo_penalty_coeffs,
        rollout_lengths,
        datasets,
        exp_name=None,
        seed=None,
        bnn_retrain_epochs=0,
        rollout_batch_size=None
    ) -> None:
        # Params always specified by user
        self.config = config
        self.dynamics_model_exps = dynamics_model_exps
        self.mopo_penalty_coeffs = mopo_penalty_coeffs
        self.rollout_lengths = rollout_lengths
        self.datasets = datasets

        # Params with defaults
        self.exp_name = exp_name
        self.seed = seed
        self.bnn_retrain_epochs = bnn_retrain_epochs
        self.rollout_batch_size = rollout_batch_size

    _jinja_template = None
    @property
    def jinja_template(self):
        if self._jinja_template is None:
            template_loader = FileSystemLoader(searchpath="./")
            env = Environment(loader=template_loader)
            self._jinja_template = env.get_template(SLURM_TRAIN_TEMPLATE_PATH)
        return self._jinja_template

    def exp_record(self, dme, mpc, rl, d):
        dynamics_exp = get_experiment_details(dme)
        return {
            "config": self.config,
            "exp_name": '_'.join(dynamics_exp.base_dir.split('_')[:-1]),
            "seed": dynamics_exp.seed,
            "dynamics_model_exp": dme,
            "bnn_retrain_epochs": self.bnn_retrain_epochs,
            "mopo_penalty_coeff": mpc,
            "rollout_length": rl,
            "rollout_batch_size": ROLLOUT_BATCH_SIZES[rl],
            "dataset": d,
        }

    @property
    def exp_collection(self):
        return (self.exp_record(*params_set) for params_set in  itertools.product(
            self.dynamics_model_exps,
            self.mopo_penalty_coeffs,
            self.rollout_lengths,
            self.datasets,
        ))

    @property
    def slurm_tmp_filename(self):
        t_stamp = datetime.utcfromtimestamp(int(time.time())).strftime(TIME_FORMAT)
        return os.path.join(SLRUM_AUTORUN_DIR, f'train.slurm.peta4-icelake.{t_stamp}')
        
    def run_experiments(self):
        failure = False
        exps_triggered = []
        for i in self.exp_collection:
            slrum_tmp_file = self.slurm_tmp_filename
            with open(slrum_tmp_file, 'w') as f:
                f.write(self.jinja_template.render(i))
            stream = os.popen(f'sbatch {slrum_tmp_file}')
            output = stream.read()
            
            if not output.startswith('Submitted batch job '):
                failure = True
                print(output)
                break

            job_id = output.split()[-1]
            exps_triggered.append(job_id)
            with open(slrum_tmp_file+'.' + str(job_id), 'w') as f:
                f.write('')
            
            time.sleep(1)

        if failure: 
            raise RuntimeError('Did not trigger job successfully!')
                
        return exps_triggered

def run_experiment_set(exp_set_dict):
    MopoExpSet(**exp_set_dict).run_experiments()

def main(params_filepath):
    with open(params_filepath, 'r') as f:
        params_collection = json.load(f)
    exps_triggered = [run_experiment_set(i) for i in params_collection]
    print(exps_triggered)

if __name__ == '__main__':
    # params_filepath = sys.argv[1]

    params_filepath = "/home/ajc348/rds/hpc-work/mopo/exp_params.170922.1.json"

    main(params_filepath)
