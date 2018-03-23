#!/bin/bash
#SBATCH --job-name=Cosmo_job
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache
#SBATCH --exclusive
#SBATCH --reservation=cosmoflow
#SBATCH -p debug
#SBATCH -A dasrepo
#DW persistentdw name=CosmoFlow2


module load /global/cscratch1/sd/djbard/cosmoML/module/March19-newCray-global

unset OMP_NUM_THREADS
export KMP_AFFINITY=compact,norespect
export KMP_HW_SUBSET=66C@2,1T


BZsize=1
i=64

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*

echo date

LD_PRELOAD=./syfix.so srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none   python CosmoNet.py > ${LOGName} 2>&1

echo date
