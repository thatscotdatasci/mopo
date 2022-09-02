#! /bin/bash

############
# MIXED-RT-3
############

for model in \
HC025 \
HC026 \
HC027 \
HC028 \
HC029 \
HC030 \
HC031 \
HC032 \
HC033 \
HC034 \
HC035 \
HC036 \
HC037 \
HC038 \
HC039 \
HC040 \
HC041 \
HC042 \
HC043 \
HC044 \
HC045 \
HC046 \
HC047 \
HC048
do
    .env/bin/python dogo/score_model_rt.py $model
done
