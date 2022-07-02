#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP437 \
MP438 \
MP439 \
MP440 \
MP441 \
MP442 \
MP443 \
MP444 \
MP445 \
MP446 \
MP447 \
MP448
do
    .env/bin/python dogo/score_model_new.py $model
done
