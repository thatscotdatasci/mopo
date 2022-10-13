#! /bin/bash

#####################
# Score Hopper models
#####################

for model in \
HO001 \
HO002 \
HO003 \
HO004 \
HO005 \
HO006 \
HO007 \
HO008 \
HO009 \
HO010 \
HO011 \
HO012
do
    .env/bin/python dogo/score_model_rt_hopper.py $model
done
