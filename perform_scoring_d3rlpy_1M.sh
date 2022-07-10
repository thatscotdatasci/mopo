#! /bin/bash

################################
# D3RLPY - RT - 1M Record Models
################################

for model in \
MP497 \
MP498 \
MP499 \
MP500 \
MP501 \
MP502 \
MP503 \
MP504 \
MP505 \
MP506 \
MP507 \
MP508 \
MP539 \
MP540 \
MP541
do
    .env/bin/python dogo/score_model_d3rlpy.py $model
done
