#! /bin/bash

############
# MIXED-RT-3
############

for model in \
WA145 \
WA146 \
WA147 \
WA148 \
WA149 \
WA150 \
WA151 \
WA152 \
WA153 \
WA154 \
WA155 \
WA156
do
    .env/bin/python dogo/score_model_rt_walker2d.py $model
done
