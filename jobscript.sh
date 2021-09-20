#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 8
#$ -l short

source /mnt/eps01-rds/jf01-home01/shared/load_DAMASK-v3a3.sh 
source activate /mnt/eps01-rds/jf01-home01/shared/.conda/damask_env
mpirun -n $NSLOTS DAMASK_grid -l tensionY.yaml -g testcase_standard.vtr
