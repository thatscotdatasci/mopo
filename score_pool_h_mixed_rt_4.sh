#! /bin/bash

##############
# H-MIXED-RT-1
##############

for model in \
HO313 \
HO314 \
HO315 \
HO316 \
HO317 \
HO318 \
HO319 \
HO320 \
HO321 \
HO322 \
HO323 \
HO324
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model --env-name Hopper
done
