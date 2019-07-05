#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import pp
import multiprocessing
import os
import argparse
import gc

gc.enable()

np.set_printoptions(threshold=np.inf)
filelist = []
for filename in Path('features').glob('**/*.files'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.bh'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.mfcc'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.mfcckl'):
    filelist.append(filename)
#for filename in Path('features').glob('**/*.rp'):
    #filelist.append(filename)
#for filename in Path('features').glob('**/*.rh'):
    #filelist.append(filename)
for filename in Path('features').glob('**/*.chroma'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.notes'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.files'):
    filelist.append(filename)

for myfile in filelist: 
    with open(str(PurePath(myfile)), 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(str(PurePath(myfile)) + ".csv", 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('ID', 'features'))
            writer.writerows(lines)
    print str(PurePath(myfile))


