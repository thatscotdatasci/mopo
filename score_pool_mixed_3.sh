#! /bin/bash

#########
# MIXED-3
#########

for model in \
MP874 \
MP875 \
MP876 \
MP877 \
MP878 \
MP879 \
MP880 \
MP881 \
MP882 \
MP883 \
MP884 \
MP885
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model
done
