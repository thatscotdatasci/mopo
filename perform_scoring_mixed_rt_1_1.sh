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
# MP759 \
# MP772 \
# MP773 \
# MP774

# First larger batch size run - unlikely to rescore
# MP775 \
# MP776 \
# MP777

# 1 Epoch of Retraining
# MP862 \
# MP863 \
# MP864 \
# MP865 \
# MP866 \
# MP867

# 1 Epoch of Retraining - D4RL MR
# MP898 \
# MP899 \
# MP900 \
# MP901 \
# MP902 \
# MP903

for model in \
MQ233 \
MQ234 \
MQ235 \
MQ236 \
MQ237 \
MQ238
do
    .env/bin/python dogo/score_model_rt.py $model
done
