#! /bin/bash

##########################
# Score HalfCheetah models
##########################

for model in \
MP760 \
MP761 \
MP762 \
MP763 \
MP764 \
MP765 \
MP766 \
MP767 \
MP768 \
MP769 \
MP770 \
MP771
do
    .env/bin/python dogo/score_model_rt.py $model
done
