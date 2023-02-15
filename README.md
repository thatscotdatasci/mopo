# Model-based Offline Policy Optimization (MOPO)

Code to reproduce the experiments in [MOPO: Model-based Offline Policy ls](https://arxiv.org/pdf/2005.13239.pdf).



## Installation
1. Install [MuJoCo 2.0](https://www.roboti.us/index.html) at `~/.mujoco/mujoco200` and copy your license key to `~/.mujoco/mjkey.txt`
2. Create a conda environment and install mopo
```
cd mopo
conda env create -f environment/gpu-env.yml
conda activate mopo
# Install viskit
git clone https://github.com/vitchyr/viskit.git
pip install -e viskit
pip install -e .
```

## Usage
Configuration files can be found in `examples/config/`. For example, run the following command to run HalfCheetah-mixed benchmark in D4RL.

```
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_mixed --gpus=1 --trial-gpus=1
```

Currently only running locally is supported.


#### Logging

This codebase contains [viskit](https://github.com/vitchyr/viskit) as a submodule. You can view saved runs with:
```
viskit ~/ray_mopo --port 6008
```
assuming you used the default [`log_dir`](examples/config/halfcheetah/0.py#L7).

## Citing MOPO
If you use MOPO for academic research, please kindly cite our paper the using following BibTeX entry.

```
@article{yu2020mopo,
  title={MOPO: Model-based Offline Policy Optimization},
  author={Yu, Tianhe and Thomas, Garrett and Yu, Lantao and Ermon, Stefano and Zou, James and Levine, Sergey and Finn, Chelsea and Ma, Tengyu},
  journal={arXiv preprint arXiv:2005.13239},
  year={2020}
}
```
# environment model
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom_rex --gpus=1 --trial-gpus=1 --checkpoint-frequency=100 --seed 1 --rex-beta=5 --exp-name rexb5 --dataset MIXED-RT-5
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom_rex --gpus=1 --trial-gpus=1 --checkpoint-frequency=100 --seed 2 --rex-beta=5 --exp-name rexb5 --dataset MIXED-RT-5
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom --gpus=1 --trial-gpus=1 --checkpoint-frequency=100 --seed 1 --rex-beta=0 --exp-name rexb0 --dataset MIXED-RT-5
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom --gpus=1 --trial-gpus=1 --checkpoint-frequency=100 --seed 2 --rex-beta=0 --exp-name rexb0 --dataset MIXED-RT-5


# agent
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom --cpus=1 --trial-cpus=1 --checkpoint-frequency=100 --exp-name halfcheetah_mixed_rt_49 --seed 1449 --dynamics-model-exp MP748 --bnn-retrain-epochs 0 --penalty-coeff 0.0 --rollout-length 5 --rollout-batch-size 50000 --dataset MIXED-RT-1

mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom --cpus=1 --trial-cpus=1 --checkpoint-frequency=100 --exp-name halfcheetah_mixed_rt_49 --seed 1449 --bnn-retrain-epochs 0 --penalty-coeff 0.0 --rollout-length 5 --rollout-batch-size 50000 --dataset MIXED-RT-1

mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_custom --cpus=1 --trial-cpus=1 --checkpoint-frequency=100 --exp-name pen0 --seed 2 --bnn-retrain-epochs 0 --penalty-coeff 0.0 --rollout-length 5 --rollout-batch-size 50000 --dataset MIXED-RT-1

