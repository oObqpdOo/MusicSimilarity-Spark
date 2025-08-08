#!/bin/bash
mpiexec --hostfile hostfile python mpi4py_rhythm.py -rp -rh ./ features0/out
