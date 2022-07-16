#! /bin/bash

########################################
# SAC - 0.1M
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
"MP674" \
"MP675" \
"MP676" \
"MP551" \
"MP552" \
"MP553" \
"MP554" \
"MP555" \
"MP556" \
"MP557" \
"MP558" \
"MP559" \
"MP560" \
"MP561" \
"MP562" \
"MP563" \
"MP564" \
"MP565"
)

for i in ${policy_experiments[@]}
do
    for j in ${dynamics_experiments[@]}
    do
        .env/bin/python examples/development/simulate_policy_dynamics_model.py --policy-experiment=$i --dynamics-experiment=$j --num-rollouts=5
    done
done
