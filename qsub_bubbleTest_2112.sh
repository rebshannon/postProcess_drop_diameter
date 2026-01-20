#!/bin/bash -l

#$ -m bea 
#$ -j y
#$ -N lessEntries
#$ -l h_rt=48:00:00
#MPI_BUFFER_SIZE = 8192
#$ -P aeracous

#  Any number <= 28:
#$ -pe omp 1

# Load modules
module load python3

conda activate work-env
export PYTHONPATH='/projectnb/aeracous/REBECCA/Silo-main/install/lib64:'${PYTHONPATH}

pip install -e .
echo 'installed, running code now...'

python new_run_getDiams.py

echo 'after python code here'
