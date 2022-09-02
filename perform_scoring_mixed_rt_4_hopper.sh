#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HO025 \
HO026 \
HO027 \
HO028 \
HO029 \
HO030 \
HO031 \
HO032 \
HO033 \
HO034 \
HO035 \
HO036
do
    .env/bin/python dogo/score_model_rt_hopper.py $model
done
