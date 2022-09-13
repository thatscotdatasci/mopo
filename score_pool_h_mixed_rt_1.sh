#! /bin/bash

##############
# H-MIXED-RT-1
##############
# HO265 \
# HO266 \
# HO267 \
# HO274 \
# HO275 \
# HO276

for model in \
HO268 \
HO269 \
HO270 \
HO271 \
HO272 \
HO273
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model --env-name Hopper
done
