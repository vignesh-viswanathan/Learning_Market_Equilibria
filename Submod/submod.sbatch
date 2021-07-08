#!/bin/bash
#
#SBATCH --job-name=pre-final-submod
#SBATCH --output=out_submod.txt  # output file
#SBATCH -e err_submod.txt       # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 

#SBATCH --time=2-20:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=9			# Number of tasks
#SBATCH --cpus-per-task=20 
#SBATCH --nodes=9

module add python/3.7.3 
which python /cm/shared/apps/python/3.7.3/bin/python
which virtualenv /cm/shared/apps/python/3.7.3/bin/virtualenv

virtualenv ~/mypython
source ~/mypython/bin/activate

pip install numpy
pip install docplex
pip install joblib

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python submod_parallel.py < case1_3.txt &
python submod_parallel.py < case1_5.txt &
python submod_parallel.py < case1_10.txt &
python submod_parallel.py < case2_3.txt &
python submod_parallel.py < case2_5.txt &
python submod_parallel.py < case2_10.txt &
python submod_parallel.py < case3_3.txt &
python submod_parallel.py < case3_5.txt &
python submod_parallel.py < case3_10.txt &
wait


