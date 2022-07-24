#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################
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

for model in \
MP874 \
MP875 \
MP876 \
MP877 \
MP878 \
MP879
do
    .env/bin/python dogo/visualisation/dynamics_model_landscape.py --dynamics-experiment=$model --dataset=MIXED-3
done
