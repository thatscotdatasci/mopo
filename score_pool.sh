#! /bin/bash

###################################################################
# Score the transitions sampled from the model pool during training
# The environment must be specified, if it is not HalfCheetah
###################################################################

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
