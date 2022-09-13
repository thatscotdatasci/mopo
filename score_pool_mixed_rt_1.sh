#! /bin/bash

#####################################
# MIXED-RT-1 - No Decay - 1 Epoch Set
#####################################
# MP862 \
# MP863 \
# MP864 \
# MP865 \
# MP866 \
# MP867 \
# MP868 \
# MP869 \
# MP870 \
# MP871 \
# MP872 \
# MP873

# MOPO Penalty 0
# MP919 \
# MP920 \
# MP921 \
# MP922 \
# MP923 \
# MP924 \
# MP925 \
# MP926 \
# MP927

# D4RL - MR Experiments
# MP898 \
# MP899 \
# MP900 \
# MP901 \
# MP902 \
# MP903 \
# MP904 \
# MP905 \
# MP906 \
# MP907 \
# MP908 \
# MP909

# D4RL - MR Experiments - 0 Additional Epochs
# MP910 \
# MP911 \
# MP912

# D4RL - MR Experiments - 3 Additional Epochs
# MP913 \
# MP914 \
# MP915 \
# MP916 \
# MP917 \
# MP918

for model in \
MP982 \
MP983 \
MP984 \
MP988 \
MP989 \
MP990
do
    .env/bin/python dogo/model_pool/score_pool.py --policy-experiment $model
done
