#! /bin/bash

#################################
# HalfCheetah MIXED-RT-1 Datasets
#################################

for model in \
MP868 \
MP869 \
MP870 \
MP871 \
MP872 \
MP873
do
    .env/bin/python dogo/visualisation/dynamics_model_landscape.py --dynamics-experiment=$model --dataset=MIXED-RT-1
done
