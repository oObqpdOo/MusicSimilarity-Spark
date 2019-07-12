#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.ml.linalg
import pyspark.mllib.param
from scipy.spatial import distance
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d

#from pyspark import SparkContext, SparkConf
#from pyspark.sql import SQLContext, Row
#confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
#confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
#sc = SparkContext(conf=confCluster)
#sqlContext = SQLContext(sc)

song = "music/Electronic/The XX - Intro.mp3"

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]




def get_neighbors_rp_euclidean(song):
    #########################################################
    #   Pre- Process RH for Euclidean
    #
    rp = sc.textFile("features/out[0-9]*.rp")
    rp = rp.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    rp = rp.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    rp = rp.map(lambda x: x.split(';'))
    rp = rp.map(lambda x: (x[0], x[1].split(",")))
    kv_rp= rp.map(lambda x: (x[0], list(x[1:])))
    rp_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #  
    comparator = rp_vec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultRH = rp_vec.map(lambda x: (x[0], distance.euclidean(x[1], comparator_value[0])))
    max_val = resultRH.max(lambda x:x[1])[1]
    min_val = resultRH.min(lambda x:x[1])[1]  
    resultRH = resultRH.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultRH 

def get_neighbors_notes(song):
    #########################################################
    #   Pre- Process Notes for Levenshtein
    #
    notes = sc.textFile("features/out[0-9]*.notes")
    notes = notes.map(lambda x: x.split(';'))
    notes = notes.map(lambda x: (x[0].replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
    notes = notes.map(lambda x: (x[0], x[3].replace(',','').replace(' ',''), x[1], x[2]))
    #########################################################
    #   Get Neighbors
    #  
    comparator = notes.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultNotes = notes.map(lambda x: (x[0], levenshtein(x[1], comparator_value[0]), x[1], x[2]))
    max_val = resultNotes.max(lambda x:x[1])[1]
    min_val = resultNotes.min(lambda x:x[1])[1]  
    resultNotes = resultNotes.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val), x[2], x[3]))  
    return resultNotes

def get_neighbors_mfcc_euclidean(song):
    #########################################################
    #   Pre- Process MFCC for Euclidean
    #
    mfcceuc = sc.textFile("features/out[0-9]*.mfcc")
    mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcceuc = mfcceuc.map(lambda x: x.split(';'))
    mfcceuc = mfcceuc.map(lambda x: (x[0], x[1].split(',')))
    mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #
    comparator = mfccVec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    resultMfcc = mfccVec.map(lambda x: (x[0], distance.euclidean(x[1], comparator_value[0])))
    max_val = resultMfcc.max(lambda x:x[1])[1]
    min_val = resultMfcc.min(lambda x:x[1])[1]  
    resultMfcc = resultMfcc.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultMfcc

def get_nearest_neighbors(song, outname):
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    mergedSim = neighbors_mfcc_eucl.leftOuterJoin(neighbors_rp_euclidean)
    mergedSim = mergedSim.leftOuterJoin(neighbors_notes)
    mergedSim = mergedSim.map(lambda x: (x[0], ((x[1][0][1] + x[1][1] + x[1][0][0]) / 3))).sortBy(lambda x: x[1], ascending = True)
    out_name = outname#"output.csv"
    #mergedSim.toPandas().to_csv(out_name, encoding='utf-8')
    return mergedSim

song = "music/Electronic/The XX - Intro.mp3"
result = get_nearest_neighbors(song, "Electro.csv")
result.sortBy(lambda x: x[1], ascending = True).take(10)


