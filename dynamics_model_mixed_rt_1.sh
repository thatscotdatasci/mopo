#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

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
# MP774

# MP862 \
# MP863 \
# MP864 \
# MP865 \
# MP866 \
# MP867

for model in \
MP868 \
MP869 \
MP870 \
MP871 \
MP872 \
MP873
do
    .env/bin/python dogo/visualisation/dynamics_model_landscape.py --dynamics-experiment=$model --dataset=MIXED-RT-1
done
