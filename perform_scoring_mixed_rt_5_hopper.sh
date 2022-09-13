#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HO145 \
HO146 \
HO147 \
HO148 \
HO149 \
HO150 \
HO151 \
HO152 \
HO153 \
HO154 \
HO155 \
HO156
do
    .env/bin/python dogo/score_model_rt_hopper.py $model
done
