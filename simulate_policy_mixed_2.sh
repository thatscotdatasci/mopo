#! /bin/bash

########################################
# MIXED-2 Hyperparameter Search MSE Exps
########################################
# All of the below represent evaluating the
# dynamics model that was use to train the
# policy.
# "MP301 MP277" \
# "MP302 MP278" \
# "MP303 MP279" \
# "MP304 MP280" \
# "MP305 MP281" \
# "MP306 MP282" \
# "MP307 MP283" \
# "MP308 MP284" \
# "MP309 MP285" \
# "MP310 MP286" \
# "MP311 MP287" \
# "MP312 MP288"

# Evaluate REx beta=10 policy on no-REx dynamics
# "MP310 MP277" \
# "MP311 MP278" \
# "MP312 MP279"

# Evaluate no-REx policy on REx beta=10 dynamics
# "MP301 MP286" \
# "MP302 MP287" \
# "MP303 MP288"

for i in \
"MP301 MP286" \
"MP302 MP287" \
"MP303 MP288"
do
    set -- $i
    .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$1 --dynamics-experiment=$2 --num-rollouts=5
done
