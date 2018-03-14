#!/bin/bash
#SBATCH --job-name=Cosmo_mlcomm
#SBATCH --time=2:00:00
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH --partition=regular
#SBATCH -C knl,quad,cache
#SBATCH --exclusive

module load /global/cscratch1/sd/djbard/cosmoML/module/March12

unset OMP_NUM_THREADS
export KMP_AFFINITY=compact,norespect
export KMP_HW_SUBSET=66C@2,1T


BZsize=1
i=160
LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))_Ep80_noBN_myData-128cube
rm -rf result/*
srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none LD_PRELOAD=./syfix.so  python CosmoNet_128.py > ${LOGName} 2>&1

