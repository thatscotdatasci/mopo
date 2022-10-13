#! /bin/bash

###############################################################
# This will run the `simulate_policy_dynamics_model.py` script, 
# using the dynamics model used in policy trainined
###############################################################

for i in \
HO325 \
HO326 \
HO327 \
HO328 \
HO329 \
HO330 \
HO331 \
HO332 \
HO333 \
HO334 \
HO335 \
HO336
do
    # Use the --stochastic-model and --stochastic-policy flags to run the dynamics model/policy stochastically - deterministic mode is the default
    .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --num-rollouts=10 --stochastic-model --seed $1
done
