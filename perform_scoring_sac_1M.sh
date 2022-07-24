#! /bin/bash

#############################
# SAC - RT - 1M Record Models
#############################

for model in \
MP683 \
MP684 \
MP685 \
MP581 \
MP582 \
MP583 \
MP584 \
MP585 \
MP586 \
MP587 \
MP588 \
MP589 \
MP590 \
MP591 \
MP592 \
MP593 \
MP594 \
MP595
do
    .env/bin/python dogo/score_model_rt.py $model
done
