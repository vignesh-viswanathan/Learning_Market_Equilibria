#!/bin/bash
#
#SBATCH --job-name=pre-final-unitdemand
#SBATCH --output=out.txt  # output file
#SBATCH -e err.txt       # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 

#SBATCH --time=2-20:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=3			# Number of tasks
#SBATCH --cpus-per-task=10 
#SBATCH --nodes=3

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

python unitdemand_parallel.py < case1.txt &
python unitdemand_parallel.py < case2.txt &
python unitdemand_parallel.py < case3.txt &
wait