#! /bin/bash

############
# MIXED-RT-3
############

for model in \
WA001 \
WA002 \
WA003 \
WA004 \
WA005 \
WA006 \
WA007 \
WA008 \
WA009 \
WA010 \
WA011 \
WA012
do
    .env/bin/python dogo/score_model_rt_walker2d.py $model
done
