#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HC001 \
HC002 \
HC003 \
HC004 \
HC005 \
HC006 \
HC007 \
HC008 \
HC009 \
HC010 \
HC011 \
HC012 \
HC013 \
HC014 \
HC015 \
HC016 \
HC017 \
HC018 \
HC019 \
HC020 \
HC021 \
HC022 \
HC023 \
HC024
do
    .env/bin/python dogo/score_model_rt.py $model
done
