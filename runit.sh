#!/bin/bash
#SBATCH --job-name=Cosmo_job
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache
#SBATCH --exclusive
#SBATCH -A dasrepo
#SBATCH --reservation=cosmoflow
#DW persistentdw name=CosmoFlow3param

module load /global/cscratch1/sd/pjm/modulefiles/cosmoflow-gb-apr10

BZsize=1
i=$SLURM_JOB_NUM_NODES

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*


LD_PRELOAD=./syfix.so srun -u -n ${i} --ntasks-per-node=1 --nodes=${i} --cpu_bind=none   python CosmoNet.py > ${LOGName} 2>&1

