#! /bin/bash

# Run the example simulation job
# Need to run this on a remote desktop
python examples/development/simulate_policy.py \
    "/home/ajc348/rds/hpc-work/mopo/runs_keep/seed:1443_2022-05-11_12-43-30uimp9cp_/checkpoint_70" \
    --max-path-length 1000 \
    --num-rollouts 1 \
    --render-mode human

# This is the format to run for the softlearning library
python -m examples.development.simulate_policy \
    "/home/ajc348/rds/hpc-work/mopo/runs_keep/seed:1443_2022-05-11_12-43-30uimp9cp_/checkpoint_70" \
    --max-path-length 1000 \
    --num-rollouts 1 \
    --render-kwargs '{"mode": "human"}'
