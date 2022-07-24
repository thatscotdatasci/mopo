#! /bin/bash

###############################
# SAC - RT - 0.1M Record Models
###############################

for model in \
MP674 \
MP675 \
MP676 \
MP551 \
MP552 \
MP553 \
MP554 \
MP555 \
MP556 \
MP557 \
MP558 \
MP559 \
MP560 \
MP561 \
MP562 \
MP563 \
MP564 \
MP565
do
    .env/bin/python dogo/score_model_rt.py $model
done
