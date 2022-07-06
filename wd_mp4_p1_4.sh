#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP413 \
MP414 \
MP415 \
MP416 \
MP417 \
MP418 \
MP419 \
MP420 \
MP421 \
MP422 \
MP423 \
MP424
do
    .env/bin/python dogo/wasserstein/wasserstein_distance_mixed_3.py $model
done
