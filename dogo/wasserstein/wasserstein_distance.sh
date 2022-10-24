#! /bin/bash

############################################################################
# Calculate the Wasserstein distances (WDs) for the passed dynamics models.
# The python script that's called specifies the datasets to be considered. 
# Note that this keeps the model constant and measures the WDs between 
# datasets. Really what we'd wanted was to measure the distances in the
# predictive distributions of different models while holding the dataset
# constant. Logic for this can be found in the wasserstein_distance.py file.
# It is necessary that each model has already been scored, such that `_means`
# and `_vars` files have been produced.
############################################################################

for model in \
MP401 \
MP402 \
MP403 \
MP404 \
MP405 \
MP406 \
MP407 \
MP408 \
MP409 \
MP410 \
MP411 \
MP412
do
    .env/bin/python dogo/wasserstein/wasserstein_distance_mixed_3.py $model
done
