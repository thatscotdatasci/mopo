#! /bin/bash

######
# D4RL
######

for model in \
MP521 \
MP522 \
MP523 \
MP524 \
MP525 \
MP526 \
MP527 \
MP528 \
MP529 \
MP530 \
MP531 \
MP532
do
    .env/bin/python dogo/score_model_d4rl.py $model
done
