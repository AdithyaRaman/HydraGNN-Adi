#!/bin/bash
#SBATCH -N 2
#SBATCH -A m4133_g
#SBATCH -J HydraGNN_MTL
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH --constrain=gpu


module reset
module load pytorch/2.0.1

HYDRAGNN_DIR=/global/cfs/cdirs/m4133/HydraGNN-gb24
module use -a /global/cfs/cdirs/m4133/jyc/perlmutter/sw/modulefiles
module load hydragnn/pytorch2.0.1-v2
echo "python:" `which python`
#export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

## Envs
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

echo "Slurm Job Nodes:$SLURM_JOB_NODES"

#srun -N1 --ntasks-per-node=4 --gpus-per-task=1 python examples/multidataset/train.py  --modelname=multi --multi_model_list="OC2022" --inputfile=SMALL_MTL.json --num_samples=3500 --num_epoch=4 --multi --ddstore --everyone
ntasks=4
model_size="SMALL"

current_datetime=$(date +'%Y-%m-%d-%H%-M-%S')
echo $current_datetime

srun -N2 -c32 --ntasks-per-node=$ntasks --gpus-per-task=1 \
     python examples/multidataset/train.py \
     --modelname=multi --multi_model_list="OC2022" --inputfile=${model_size}_MTL.json --num_samples=3500 \
        --num_epoch=4 --multi --ddstore --everyone
