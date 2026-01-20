#!/bin/bash -l

#$ -m bea 
#$ -j y
#$ -N sembEllipse
#$ -l h_rt=12:00:00
#MPI_BUFFER_SIZE = 8192
#$ -P aeracous

#  Any number <= 28:
#$ -pe omp 1

# Load modules
module load python3

conda activate work-env
export PYTHONPATH='/projectnb/aeracous/REBECCA/Silo-main/install/lib64:'${PYTHONPATH}

cd $AHOME/postProcess_drop_diameter/

pip install -e .
echo 'installed, running code now...'
echo 'conda prefix is'$CONDA_PREFIX
python of_getDiams.py

echo 'after python code here'
