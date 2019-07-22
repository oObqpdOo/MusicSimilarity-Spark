#!/bin/bash
#SBATCH --partition=s_standard
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p shared

MAX = 36 as per wiki https://ara-wiki.rz.uni-jena.de/index.php?title=Hauptseite
#SBATCH -n 36

Timeout
#SBATCH --time=04:00:00

32 cores ( could do 74 with hyperthreading)
#SBATCH -c 72

Memory per CORE - 192GB / 32 Cores - max mem
#SBATCH --mem-per-cpu=2000

module load nvidia/cuda/10.1.168
module load compiler/gcc/7.3.0
module load libs/blas/intel-2018
module load libs/lapack/intel-2018
module load tools/python/2.7
module load mpi/openmpi/3.1.2-gcc-7.3.0

srun -n $SLURM_NTASKS --mpi=openmpi python mpi4py_test.py

