#! /bin/bash

###############################
# MIXED-2 Hyperparameter Search
###############################

for model in \
MP277 \
MP278 \
MP279 \
MP280 \
MP281 \
MP282 \
MP283 \
MP284 \
MP285 \
MP286 \
MP287 \
MP288 \
MP289 \
MP290 \
MP291 \
MP292 \
MP293 \
MP294 
do
    .env/bin/python dogo/score_model_mixed_2.py $model
done
