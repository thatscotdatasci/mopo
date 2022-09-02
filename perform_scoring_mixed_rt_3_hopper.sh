#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HO013 \
HO014 \
HO015 \
HO016 \
HO017 \
HO018 \
HO019 \
HO020 \
HO021 \
HO022 \
HO023 \
HO024
do
    .env/bin/python dogo/score_model_rt_hopper.py $model
done
