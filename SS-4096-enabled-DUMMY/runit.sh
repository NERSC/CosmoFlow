#!/bin/bash
#SBATCH --job-name=Cosmo_job
#SBATCH --time=01:00:00
#SBATCH --nodes=4096
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache
#SBATCH --exclusive
#SBATCH --reservation=CosmoFlow2_knl
###SBATCH -p special
#SBATCH -A dasrepo
####DW persistentdw name=CosmoFlowMarch20th


module load /global/cscratch1/sd/djbard/cosmoML/module/March19-newCray-global

unset OMP_NUM_THREADS
export KMP_AFFINITY=compact,norespect
export KMP_HW_SUBSET=66C@2,1T


BZsize=1
i=4096

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*

echo date

LD_PRELOAD=./syfix.so srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none   python CosmoNet.py > ${LOGName} 2>&1

echo date
