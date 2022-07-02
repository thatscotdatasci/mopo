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
    .env/bin/python dogo/score_model_new.py $model
done
