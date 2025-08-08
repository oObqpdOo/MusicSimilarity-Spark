#!/bin/bash
#SBATCH --partition=s_hadoop
#SBATCH -J download_6M
#SBATCH -o download_6M.out
#SBATCH -e download_6M.err
#SBATCH --time=8-00:00:00
#SBATCH -n 16
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=10000

module purge
module load tools/python/3.7
module load compiler/gcc/7.3.0
module load libs/blas/intel-2018
module load libs/lapack/intel-2018
module load mpi/openmpi/3.1.2-gcc-7.3.0

export http_proxy="http://internet4nzm.rz.uni-jena.de:3128"
export https_proxy="http://internet4nzm.rz.uni-jena.de:3128"

pip install mpi4py --user
pip install essentia --user

srun -n $SLURM_NTASKS --mpi=openmpi python3 download.py
