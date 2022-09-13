#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################
# Original runs
# MP329 \
# MP330 \
# MP331 \
# MP332 \
# MP333 \
# MP334 \
# MP335 \
# MP336 \
# MP337 \
# MP338 \
# MP339 \
# MP340 \
# MP723 \
# MP724 \
# MP725

# 1 Epoch of Retraining
# MP874 \
# MP875 \
# MP876 \
# MP877 \
# MP878 \
# MP879

for model in \
MP874 \
MP875 \
MP876 \
MP877 \
MP878 \
MP879
do
    .env/bin/python dogo/score_model_rt.py $model
done
