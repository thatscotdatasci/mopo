#! /bin/bash

##################################
# D3RLPY - RT - 0.1M Record Models
##################################

for model in \
MP473 \
MP474 \
MP475 \
MP476 \
MP477 \
MP478 \
MP479 \
MP480 \
MP481 \
MP482 \
MP483 \
MP484 \
MP536 \
MP537 \
MP538
do
    .env/bin/python dogo/score_model_rt.py $model
done
