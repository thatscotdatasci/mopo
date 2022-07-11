#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################
# All of the below represent evaluating the
# dynamics model that was use to train the
# policy.
# "MP365 MP329" \
# "MP366 MP330" \
# "MP367 MP331" \
# "MP368 MP332" \
# "MP369 MP333" \
# "MP370 MP334" \
# "MP371 MP335" \
# "MP372 MP336" \
# "MP373 MP337" \
# "MP374 MP338" \
# "MP375 MP339" \
# "MP376 MP340"

# Evaluate REx beta=10 policy on no-REx dynamics
# "MP374 MP329" \
# "MP375 MP330" \
# "MP376 MP331"

# Evaluate no-REx policy on REx beta=10 dynamics
# "MP365 MP338" \
# "MP366 MP339" \
# "MP367 MP340"

for i in \
"MP365 MP338" \
"MP366 MP339" \
"MP367 MP340"
do
    set -- $i
    .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$1 --dynamics-experiment=$2 --num-rollouts=5
done
