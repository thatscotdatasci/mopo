#! /bin/bash

#########
# MIXED-3
#########

for model in \
MP886 \
MP887 \
MP888 \
MP889 \
MP890 \
MP891
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model
done
