#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HC139 \
HC140 \
HC141 \
HC142 \
HC143 \
HC144 \
HC145 \
HC146 \
HC147 \
HC148 \
HC149 \
HC150
do
    .env/bin/python dogo/score_model_rt.py $model
done
