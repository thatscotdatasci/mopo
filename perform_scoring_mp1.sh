#! /bin/bash

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
    .env/bin/python dogo/score_model.py $model
done
