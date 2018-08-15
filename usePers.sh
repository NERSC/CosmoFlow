#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:05:00
#DW persistentdw name=CosmoFlow3param
####DW stage_in source=/global/cscratch1/sd/djbard/cosmoML/data-3param/8000-nodupes destination=$DW_PERSISTENT_STRIPED_CosmoFlow3param/8000-nodupes type=directory

ls -larth $DW_PERSISTENT_STRIPED_CosmoFlow3param

fix_perms -g djbard $DW_PERSISTENT_STRIPED_CosmoFlow3param/*

ls -lartSh $DW_PERSISTENT_STRIPED_CosmoFlow3param/13000-2xDupe/ 

echo "  "

ls -lartSh $DW_PERSISTENT_STRIPED_CosmoFlow3param/13000-2xDupe/val 

echo "  "

ls -lartSh $DW_PERSISTENT_STRIPED_CosmoFlow3param/13000-2xDupe/train | wc

echo "  "


#ls -lartSh $DW_PERSISTENT_STRIPED_CosmoFlow3param/8000-nodupes/test

echo "  "

