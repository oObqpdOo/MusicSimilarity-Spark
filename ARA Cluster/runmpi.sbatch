#!/bin/bash
#SBATCH --partition=s_standard
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p shared

MAX = 36 as per wiki https://ara-wiki.rz.uni-jena.de/index.php?title=Hauptseite
https://slurm.schedmd.com/cpu_management.html - tasks for the job
-n = TASKS - usually 1 task per node - in this case: n = Nodes * CPUs = 36 * 36 or 36 * 72
#SBATCH -n 1296


MAX = 36 as per wiki https://ara-wiki.rz.uni-jena.de/index.php?title=Hauptseite
-N = NODES - hard requests 36 nodes
#SBATCH -N 36

-c equals --cpus-per-task ! per TASK not per NODE - per node = #avail cpus anyway
#SBATCH --cpus-per-task=2 
#SBATCH --ntasks-per-core=1

this allows to distribute tasks without knowledge of cpu count
#SBATCH --ntasks-per-node=36
 
#SBATCH --mem-per-cpu=5000


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

