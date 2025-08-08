#!/bin/bash
#SBATCH --partition=s_hadoop
#SBATCH -J mpi4py_features
#SBATCH -o mpi4py_features.out
#SBATCH -e mpi4py_features.err
#SBATCH --time=8-00:00:00
#SBATCH -n 288
#SBATCH -N 16
#SBATCH --ntasks-per-node=18
#SBATCH --mem-per-cpu=10000

module purge
module load tools/python/3.7
module load compiler/gcc/7.3.0
module load libs/blas/intel-2018
module load libs/lapack/intel-2018
module load mpi/openmpi/3.1.2-gcc-7.3.0

pip install mpi4py --user
pip install essentia --user

srun -n $SLURM_NTASKS --mpi=openmpi python3 test_mars_extractor.py

