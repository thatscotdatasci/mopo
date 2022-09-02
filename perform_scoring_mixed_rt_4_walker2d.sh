#! /bin/bash

############
# MIXED-RT-3
############

for model in \
WA025 \
WA026 \
WA027 \
WA028 \
WA029 \
WA030 \
WA031 \
WA032 \
WA033 \
WA034 \
WA035 \
WA036
do
    .env/bin/python dogo/score_model_rt_walker2d.py $model
done
