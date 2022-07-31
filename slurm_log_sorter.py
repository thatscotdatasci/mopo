import os
import re
import sys
import json
from glob import glob

def main(results_dir: str):    
    ########################
    # Experiment State Files
    ########################
    es_paths = glob(os.path.join(results_dir, 'experiment_state-*.json'))
    for es_path in es_paths:
        with open(es_path, 'r') as f:
            exp_state = json.load(f)

        es_logdir = os.path.abspath(exp_state['checkpoints'][-1]['logdir'])

        os.rename(es_path, os.path.join(es_logdir, os.path.basename(es_path)))
    
    ############
    # SLURM Logs
    ############
    train_log_paths = glob(os.path.join('slurm_logs', 'train-log.*'))
    for train_log_path in train_log_paths:
        train_log_path = os.path.abspath(train_log_path)

        with open(train_log_path, 'r') as f:
            log_dir_res = re.findall("(?<=\[ MOPO \] log_dir: ).*(?= \|)", f.read())
            if len(log_dir_res) == 0:
                print(f'Could not find log_dir in {train_log_path}')
                continue
            log_dir = log_dir_res[0]

        if os.path.dirname(log_dir) != os.path.abspath(results_dir):
            # Only move the log files which match the directory of interest
            continue

        job_id = os.path.basename(train_log_path).split('.')[-1]
        train_log_name = f'train-log.{job_id}'
        machine_file_name = f'machine.file.{job_id}'
        slurm_file_name = f'slurm-{job_id}.out'

        os.rename(os.path.join('slurm_logs', train_log_name), os.path.join(log_dir, train_log_name))
        os.rename(machine_file_name, os.path.join(log_dir, machine_file_name))
        os.rename(slurm_file_name, os.path.join(log_dir, slurm_file_name))

    ######################################################
    # Check that each folder has exactly one expected file
    ######################################################
    exp_results_dirs = glob(os.path.join(results_dir, 'seed:*'))
    for exp_results_dir in exp_results_dirs:
        assert len(glob(os.path.join(exp_results_dir, 'train-log.*'))) == 1
        assert len(glob(os.path.join(exp_results_dir, 'machine.file.*'))) == 1
        assert len(glob(os.path.join(exp_results_dir, 'slurm-*.out'))) == 1

if __name__ == "__main__":
    # results_dir = sys.argv[1]
    results_dir = os.path.abspath('ray_mopo/HalfCheetah/halfcheetah_mixed_rt_1_101e3')
    main(results_dir)
