#!/bin/bash
#SBATCH -N 1 -C knl -t 02:00:00 -p regular
module load tensorflow
date
time python CosmoNet_noFeed.py
date
