#!/bin/bash
#SBATCH --job-name=Cosmo_mlcomm
#SBATCH --time=4:00:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH --partition=regular
#SBATCH -C knl,quad,cache
#SBATCH --exclusive

module load /global/cscratch1/sd/djbard/cosmoML/module/March12

#ulimit -m unlimited

export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export OMP_NUM_THREADS=66 ###important!!!

BZsize=1
i=512
LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))_Ep80_noBN_myData
rm -rf result/*
srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none python CosmoNet_noFeed.py > ${LOGName} 2>&1

