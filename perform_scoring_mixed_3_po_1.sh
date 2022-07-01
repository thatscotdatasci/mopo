#! /bin/bash

#############################################
# MIXED-3-PO-1 Hyperparameter Search MSE Exps
#############################################

for model in \
MP341 \
MP342 \
MP343 \
MP344 \
MP345 \
MP346 \
MP347 \
MP348 \
MP349 \
MP350 \
MP351 \
MP352
do
    .env/bin/python dogo/score_model.py $model
done
