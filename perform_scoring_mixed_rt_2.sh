#! /bin/bash

############
# MIXED-RT-2
############

for model in \
MP805 \
MP806 \
MP807 \
MP808 \
MP809 \
MP810 \
MP811 \
MP812 \
MP813 \
MP814 \
MP815 \
MP816
do
    .env/bin/python dogo/score_model_rt.py $model
done
