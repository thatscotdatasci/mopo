#! /bin/bash

#############################################
# MIXED-3-PO-2 Hyperparameter Search MSE Exps
#############################################

for model in \
MP353 \
MP354 \
MP355 \
MP356 \
MP357 \
MP358 \
MP359 \
MP360 \
MP361 \
MP362 \
MP363 \
MP364
do
    .env/bin/python dogo/score_model_po_2.py $model
done
