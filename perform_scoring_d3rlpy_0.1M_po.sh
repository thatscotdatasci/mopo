#! /bin/bash

##################################
# D3RLPY - RT - 0.1M Record Models
##################################

for model in \
MP733 \
MP734 \
MP735 \
MP736 \
MP737 \
MP738 \
MP739 \
MP740 \
MP741 \
MP742 \
MP743 \
MP744 \
MP745 \
MP746 \
MP747
do
    .env/bin/python dogo/score_model_d3rlpy_po.py $model
done
