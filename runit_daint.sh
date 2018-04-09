#!/bin/bash
#SBATCH --job-name=CosmoFlow
#SBATCH --time=01:00:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=g107
##DW jobdw capacity=3.5TB access_mode=striped type=scratch
##DW stage_in source=/scratch/snx3000/pjm/CosmoFlow_data/5000 destination=$DW_JOB_STRIPED/5000 type=directory

#ls -l $DW_JOB_STRIPED

#ls -l $DW_JOB_STRIPED/5000

#du -sh $DW_JOB_STRIPED/5000/*

#ls $DW_JOB_STRIPED/5000/val/ | wc -l
#ls -lS $DW_JOB_STRIPED/5000/train/

module load /scratch/snx3000/pjm/modulefiles/cosmoflow

ulimit -s unlimited

#exit

# set local batch size and job size for the log (does not get forwared to app in anyway)
BZsize=1
i=512

LOGName=LogCosmoFlow_N$(( ${i} ))BZ$(( ${BZsize} ))
rm -rf result/*

srun -n ${i} --cpu_bind=none -u python CosmoNet.py > ${LOGName} 2>&1
