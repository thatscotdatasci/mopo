#! /bin/bash

########################################
# SAC - 0.1M
########################################

policy_experiments=(\
"MP461" \
"MP462" \
"MP463" \
"MP464" \
"MP465" \
"MP466" \
"MP467" \
"MP468" \
"MP469" \
"MP470" \
"MP471" \
"MP472" \
"MP542" \
"MP543" \
"MP544"
)

dynamics_experiments=(\
"MP449" \
"MP450" \
"MP451" \
"MP452" \
"MP453" \
"MP454" \
"MP455" \
"MP456" \
"MP457" \
"MP458" \
"MP459" \
"MP460" \
"MP533" \
"MP534" \
"MP535"
)

for i in ${policy_experiments[@]}
do
    for j in ${dynamics_experiments[@]}
    do
        .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --dynamics-experiment=$j --num-rollouts=5
    done
done
