#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP329 \
MP330 \
MP331 \
MP332 \
MP333 \
MP334 \
MP335 \
MP336 \
MP337 \
MP338 \
MP339 \
MP340 \
MP723 \
MP724 \
MP725
do
    .env/bin/python dogo/visualisation/dynamics_model_landscape.py --dynamics-experiment=$model --dataset=D4RL-HC-MR_100000
done
