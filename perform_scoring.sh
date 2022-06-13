#! /bin/bash

##################
# PAP5 Experiments
##################
# MP100 \
# MP101 \
# MP102 \
# MP103 \
# MP104 \
# MP105 \
# MP106 \
# MP107 \
# MP108 \
# MP109 \
# MP110 \
# MP111 \
# MP124 \
# MP125 \
# MP126 \
# MP134 \
# MP135 \
# MP136 \
# MP137 \
# MP138 \
# MP139 \
# MP146 \
# MP147

#################
# MP1 Experiments
#################

for model in \
MP112 \
MP113 \
MP114 \
MP115 \
MP116 \
MP117 \
MP118 \
MP119 \
MP120 \
MP121 \
MP122 \
MP123 \
MP127 \
MP128 \
MP129 \
MP130 \
MP131 \
MP132 \
MP133 \
MP140 \
MP141 \
MP142 \
MP143 \
MP144 \
MP145
do
    python dogo/score_model.py $model
done
