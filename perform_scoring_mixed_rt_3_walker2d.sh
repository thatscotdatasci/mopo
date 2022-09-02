#! /bin/bash

############
# MIXED-RT-3
############

for model in \
WA013 \
WA014 \
WA015 \
WA016 \
WA017 \
WA018 \
WA019 \
WA020 \
WA021 \
WA022 \
WA023 \
WA024
do
    .env/bin/python dogo/score_model_rt_walker2d.py $model
done
