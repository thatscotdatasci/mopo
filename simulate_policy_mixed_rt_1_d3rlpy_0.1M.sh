#! /bin/bash

########################################
# D3RLPY - 0.1M Policies
# MIXED-RT-1 Dynamics
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
"MP748" \
"MP749" \
"MP750" \
"MP772" \
"MP773" \
"MP774" \
"MP757" \
"MP758" \
"MP759"
)

for i in ${policy_experiments[@]}
do
    for j in ${dynamics_experiments[@]}
    do
        .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --dynamics-experiment=$j --num-rollouts=5
    done
done
