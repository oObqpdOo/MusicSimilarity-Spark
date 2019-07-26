#!/bin/bash
#SBATCH --partition=s_standard
#SBATCH -J mpi4py_rhythm
#SBATCH -o mpi4py_rhythm.out
#SBATCH -e mpi4py_rhythm.err
#SBATCH --time=02:00:00
#SBATCH -n 648
#SBATCH -N 36
#SBATCH --cpus-per-task=4 
#SBATCH --ntasks-per-node=18
#SBATCH --mem-per-cpu=2500

module load nvidia/cuda/10.1.168
module load compiler/gcc/7.3.0
module load libs/blas/intel-2018
module load libs/lapack/intel-2018
module load tools/python/2.7
module load mpi/openmpi/3.1.2-gcc-7.3.0

srun -n $SLURM_NTASKS --mpi=openmpi python mpi4py_ara_rhythm.py -rp -rh /beegfs/ja62lel/fma_full features0/out
