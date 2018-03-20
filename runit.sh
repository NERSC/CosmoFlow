#!/bin/bash
#SBATCH --job-name=Cosmo_job
#SBATCH --time=04:00:00
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH --partition=regular
#SBATCH -C knl,quad,cache
#SBATCH --exclusive
#####SBATCH -q premium
#SBATCH -A dasrepo

module load /global/cscratch1/sd/djbard/cosmoML/module/March12-newCray

unset OMP_NUM_THREADS
export KMP_AFFINITY=compact,norespect
export KMP_HW_SUBSET=66C@2,1T


BZsize=1
i=160

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*


LD_PRELOAD=./syfix.so srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none   python CosmoNet.py > ${LOGName} 2>&1

