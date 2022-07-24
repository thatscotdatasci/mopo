#! /bin/bash

##################################
# D3RLPY - RT - 0.1M Record Models
##################################

for model in \
MP449 \
MP450 \
MP451 \
MP452 \
MP453 \
MP454 \
MP455 \
MP456 \
MP457 \
MP458 \
MP459 \
MP460 \
MP533 \
MP534 \
MP535
do
    .env/bin/python dogo/score_model_rt.py $model
done
