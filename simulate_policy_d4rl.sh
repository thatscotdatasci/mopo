#! /bin/bash

########################################
# D4RL
########################################
# "MP521" \
# "MP522" \
# "MP523" \
# "MP524" \

policy_experiments=(\
"MP525" \
"MP526" \
"MP527" \
"MP528" \
"MP529" \
"MP530" \
"MP531" \
"MP532" \
)

dynamics_experiments=(\
"MP521" \
"MP522" \
"MP523" \
"MP524" \
"MP525" \
"MP526" \
"MP527" \
"MP528" \
"MP529" \
"MP530" \
"MP531" \
"MP532" \
)

for i in ${policy_experiments[@]}
do
    for j in ${dynamics_experiments[@]}
    do
        .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --dynamics-experiment=$j --num-rollouts=5
    done
done
