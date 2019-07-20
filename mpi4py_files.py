#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import multiprocessing
import os
import argparse
import gc

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

gc.enable()
filelist = []
for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)  
print("length of filelist" + str(len(filelist)))

def parallel_python_process(process_id, cpu_filelist):
    print("calling rank " + str(rank) + " size " + str(size))
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        with open("features0/out" + str(process_id) + ".files", "a") as myfile:
            #print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist))) 
            line = (filename + "     :       " + str(process_id))
            myfile.write(line + '\n')       
            myfile.close()
        count = count + 1
        gc.enable()
        gc.collect()
    gc.enable()
    gc.collect()
    return 1

def process_stuff(startjob, maxparts, batchsz, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):

    startjob = int(startjob)
    maxparts = int(maxparts) + 1
    files_per_part = int(batchsz)

    print("starting with: ")    
    print(startjob)
    print("ending with: ")
    print(maxparts - 1)
    # Divide the task into subtasks - such that each subtask processes around 25 songs
    print("files per part: ")
    print(files_per_part)
    
    start = 0
    end = len(filelist)
    print("used cores: " + str(size))
    ncpus = size

    parts = (len(filelist) / files_per_part) + 1
    print("Split problem in parts: ")
    print(str(parts))
    step = (end - start) / parts + 1
    if maxparts > parts:
        maxparts = parts
    for index in xrange(startjob + rank, maxparts, size):
        if index < parts:        
            starti = start+index*step
            endi = min(start+(index+1)*step, end)
            print("calling process  " + str(rank) + " index " + str(index) + " size " + str(size) + " starti " + str(starti) + " endi " + str(endi))
            with open("features0/out" + str(index) + ".files", "w") as myfile:
                myfile.write("")
                myfile.close()
            parallel_python_process(index, filelist[starti:endi])
            gc.collect()
    gc.enable()
    gc.collect()

do_mfcc_kl = 1
do_mfcc_euclid = 1
do_notes = 1
do_chroma = 1
do_bh = 1
startbatch = 2
endbatch = 2
batchsize = 25

# BATCH FEATURE EXTRACTION:
process_stuff(startbatch, endbatch, batchsize, do_mfcc_kl, do_mfcc_euclid, do_notes, do_chroma, do_bh)

