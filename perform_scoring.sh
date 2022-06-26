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
# MP112 \
# MP113 \
# MP114 \
# MP115 \
# MP116 \
# MP117 \
# MP118 \
# MP119 \
# MP120 \
# MP121 \
# MP122 \
# MP123 \
# MP127 \
# MP128 \
# MP129 \
# MP130 \
# MP131 \
# MP132 \
# MP133 \
# MP140 \
# MP141 \
# MP142 \
# MP143 \
# MP144 \
# MP145

###############################
# PAP5 Extended No REx Training
###############################
# MP162 \
# MP163 \
# MP164

#####################################
# PAP5 Hyperparameter Search MSE Exps
#####################################
# MP165 \
# MP166 \
# MP167 \
# MP168 \
# MP169 \
# MP170 \
# MP171 \
# MP172 \
# MP173 \
# MP174 \
# MP175 \
# MP176 \
# MP183 \
# MP184 \
# MP185 \
# MP186 \
# MP187 \
# MP188 \
# MP199 \
# MP200 \
# MP201
# MP232 \
# MP233 \
# MP234 \
# MP235 \
# MP236 \
# MP237 \
# MP238 \
# MP239 \
# MP241 \
# MP242 \
# MP243

####################################
# PAP5 Hyperparameter Search LL Exps
####################################
# MP193 \
# MP194 \
# MP195 \
# MP196 \
# MP197 \
# MP198

########################################
# MIXED-1 Hyperparameter Search MSE Exps
########################################
# MP202 \
# MP203 \
# MP204 \
# MP205 \
# MP206 \
# MP207 \
# MP208 \
# MP209 \
# MP210 \
# MP211 \
# MP212 \
# MP213 \
# MP214 \
# MP215 \
# MP216 \
# MP217 \
# MP218 \
# MP219 \
# MP220 \
# MP221 \
# MP222
# MP295 \
# MP296 \
# MP297 \
# MP298 \
# MP299 \
# MP300

###############################
# MIXED-2 Hyperparameter Search
###############################
# MP277 \
# MP278 \
# MP279 \
# MP280 \
# MP281 \
# MP282 \
# MP283 \
# MP284 \
# MP285 \
# MP286 \
# MP287 \
# MP288 \
# MP289 \
# MP290 \
# MP291 \
# MP292 \
# MP293 \
# MP294

for model in \
MP295 \
MP296 \
MP297 \
MP298 \
MP299 \
MP300
do
    .env/bin/python dogo/score_model.py $model
done
