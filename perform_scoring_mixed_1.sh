#! /bin/bash

########################################
# MIXED-1 Hyperparameter Search MSE Exps
########################################

for model in \
MP202 \
MP203 \
MP204 \
MP205 \
MP206 \
MP207 \
MP208 \
MP209 \
MP210 \
MP211 \
MP212 \
MP213 \
MP214 \
MP215 \
MP216 \
MP217 \
MP218 \
MP219 \
MP220 \
MP221 \
MP222 \
MP295 \
MP296 \
MP297 \
MP298 \
MP299 \
MP300
do
    .env/bin/python dogo/score_model.py $model
done
