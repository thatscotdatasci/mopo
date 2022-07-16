#! /bin/bash

#########
# MIXED-3
#########
# "MP365" \
# "MP366" \
# "MP367" \
# "MP368" \
# "MP369" \
# "MP370" \
# "MP371" \
# "MP372" \
# "MP373" \
# "MP374" \
# "MP375" \
# "MP376" \
# "MP726" \

policy_experiments=(\
"MP727" \
"MP728" \
)

dynamics_experiments=(\
"MP329" \
"MP330" \
"MP331" \
"MP332" \
"MP333" \
"MP334" \
"MP335" \
"MP336" \
"MP337" \
"MP338" \
"MP339" \
"MP340" \
"MP723" \
"MP724" \
"MP725" \
)

for i in ${policy_experiments[@]}
do
    for j in ${dynamics_experiments[@]}
    do
        .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --dynamics-experiment=$j --num-rollouts=5
    done
done
