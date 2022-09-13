#! /bin/bash

##############
# H-MIXED-RT-1
##############

for model in \
HO217 \
HO218 \
HO219 \
HO220 \
HO221 \
HO222 \
HO223 \
HO224 \
HO225 \
HO226 \
HO227 \
HO228
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model --env-name Hopper
done
