# Use the following command to request a single CPU node and one CPU
sintr -t 10:0:0 -N1 -n1 -A KRUEGER-SL3-CPU -p icelake

conda deactivate
conda activate /rds/project/rds-eWkDxBhxBrQ/dimorl/code/mopo/.env
cd /rds/project/rds-eWkDxBhxBrQ/dimorl/code/mopo
export CPATH=$CONDA_PREFIX/include
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export TF_FORCE_GPU_ALLOW_GROWTH=true
export RAY_DISABLE_MEMORY_MONITOR=1

jupyter notebook --no-browser --ip=* --port=8081
ssh -L 8081:cpu-q-28:8081 -fN hpc

