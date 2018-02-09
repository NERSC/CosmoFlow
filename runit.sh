#!/bin/bash
#SBATCH --job-name=mlcomm
#SBATCH --time=04:00:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH --partition=regular
#SBATCH -C knl,quad,cache
#SBATCH --exclusive

module load tensorflow/intel-head-MKL-DNN
module load ~/tmp_inst/modulefiles/craype-ml-plugin-py2/1.1.0

ulimit -m unlimited

export OMP_NUM_THREADS=66

for i in 1 2 4 8 16 32 64; do

rm -rf result/*
srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none python CosmoNet_noFeed.py | tee ${i}.log

sleep 5

done

exit
