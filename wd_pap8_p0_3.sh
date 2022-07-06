#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP425 \
MP426 \
MP427 \
MP428 \
MP429 \
MP430 \
MP431 \
MP432 \
MP433 \
MP434 \
MP435 \
MP436
do
    .env/bin/python dogo/wasserstein/wasserstein_distance_mixed_3.py $model
done
