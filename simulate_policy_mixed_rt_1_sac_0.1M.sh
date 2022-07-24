#! /bin/bash

########################################
# SAC - 0.1M Policies
# MIXED-RT-1 Dynamics
########################################

policy_experiments=(\
"MP677" \
"MP678" \
"MP679" \
"MP566" \
"MP567" \
"MP568" \
"MP569" \
"MP570" \
"MP571" \
"MP572" \
"MP573" \
"MP574" \
"MP575" \
"MP576" \
"MP577" \
"MP578" \
"MP579" \
"MP580"
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
