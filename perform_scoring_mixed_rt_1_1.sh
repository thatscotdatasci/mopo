#! /bin/bash

#####################################
# MIXED-RT-1 - No Decay - 1 Epoch Set
#####################################
# MP748 \
# MP749 \
# MP750 \
# MP751 \
# MP752 \
# MP753 \
# MP754 \
# MP755 \
# MP756 \
# MP757 \
# MP758 \
# MP759

# First larger batch size run - unlikely to rescore
# MP772 \
# MP773 \
# MP774

for model in \
MP775 \
MP776 \
MP777
do
    .env/bin/python dogo/score_model_rt.py $model
done
