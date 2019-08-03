#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path, PurePath
import shutil
import os
import argparse

pathname = './'

#===================================================================
# FILES
#===================================================================


extension = ".files"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# BH
#===================================================================

extension = ".bh"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# MFCC
#===================================================================

extension = ".mfcc"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# MFCCKL
#===================================================================

extension = ".mfcckl"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# RP
#===================================================================

extension = ".rp"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# RH
#===================================================================

extension = ".rh"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# CHROMA
#===================================================================

extension = ".chroma"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))

#===================================================================
# NOTES
#===================================================================

extension = ".notes"
filelist = []

for filename in Path(pathname).glob('**/*' + extension):
    filelist.append(filename)

with open("out" + extension, 'wb') as wfd:
    for f in filelist:
        with open(str(f),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
        print str(PurePath(f))




