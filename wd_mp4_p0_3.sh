#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP401 \
MP402 \
MP403 \
MP404 \
MP405 \
MP406 \
MP407 \
MP408 \
MP409 \
MP410 \
MP411 \
MP412
do
    .env/bin/python dogo/wasserstein/wasserstein_distance_mixed_3.py $model
done
