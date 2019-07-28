#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import os
import argparse
import gc

gc.enable()

pathname = 'features0'

np.set_printoptions(threshold=np.inf)
filelist = []
for filename in Path(pathname).glob('**/*.files'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.bh'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.mfcc'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.mfcckl'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.rp'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.rh'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.chroma'):
    filelist.append(filename)
for filename in Path(pathname).glob('**/*.notes'):
    filelist.append(filename)

for myfile in filelist: 
    with open(str(PurePath(myfile)), 'r') as file :
        filedata = file.read()

    filedata = filedata.replace("/beegfs/ja62lel/fma_full/", "")
    filedata = filedata.replace("/beegfs/ja62lel/private/", "")

    # Write the file out again
    with open(str(PurePath(myfile)), 'w') as file:
        file.write(filedata)

    print str(PurePath(myfile))
