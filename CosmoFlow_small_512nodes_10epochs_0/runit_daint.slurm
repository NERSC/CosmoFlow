#!/bin/bash
#SBATCH --job-name=CosmoFlow
#SBATCH --time=02:00:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=g107

module load /scratch/snx3000/pjm/modulefiles/cosmoflow

ulimit -s unlimited

# set local batch size and job size for the log (does not get forwared to app in anyway)
BZsize=1
i=512

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*

srun -n ${i} --cpu_bind=none -u python CosmoNet.py > ${LOGName} 2>&1
