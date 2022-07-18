#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP748 \
MP749 \
MP750 \
MP751 \
MP752 \
MP753 \
MP754 \
MP755 \
MP756 \
MP757 \
MP758 \
MP759 \
MP772 \
MP773 \
MP774
do
    .env/bin/python dogo/visualisation/dynamics_model_landscape.py --dynamics-experiment=$model --dataset=SAC-RT-3M-2-P0_10000
done
