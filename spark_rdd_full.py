#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.mllib.linalg
from scipy.spatial import distance
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)

def chroma_cross_correlate(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([length1,12])
    chroma1 = chroma1_par.reshape(length1, 12)
    length2 = chroma2_par.size/12
    chroma2 = np.empty([length2,12])
    chroma2 = chroma2_par.reshape(length2, 12)
    corr = sp.signal.correlate2d(chroma1, chroma2, mode='full') 
    transposed_chroma = np.transpose(corr)
    mean_line = transposed_chroma[12]
    #print np.max(mean_line)
    return np.max(mean_line)

def chroma_cross_correlate_full(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([length1,12])
    chroma1 = chroma1_par.reshape(length1, 12)
    length2 = chroma2_par.size/12
    chroma2 = np.empty([length2,12])
    chroma2 = chroma2_par.reshape(length2, 12)
    corr = sp.signal.correlate2d(chroma1, chroma2, mode='full')
    #transposed_chroma = np.transpose(transposed_chroma)
    #mean_line = transposed_chroma[12]
    #print np.max(corr)
    return np.max(corr)

def chroma_cross_correlate_valid(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([length1,12])
    length2 = chroma2_par.size/12
    chroma2 = np.empty([length2,12])
    if(length1 > length2):
        chroma1 = chroma1_par.reshape(length1, 12)
        chroma2 = chroma2_par.reshape(length2, 12)
    else:
        chroma2 = chroma1_par.reshape(length1, 12)
        chroma1 = chroma2_par.reshape(length2, 12)    
    corr = sp.signal.correlate2d(chroma1, chroma2, mode='same')
    transposed_chroma = corr.transpose()  
    transposed_chroma = transposed_chroma / (min(length1, length2))
    transposed_chroma = transposed_chroma.transpose()
    transposed_chroma = np.transpose(transposed_chroma)
    mean_line = transposed_chroma[6]
    #print np.max(mean_line)
    return np.max(mean_line)

#get 13 mean and 13x13 cov as vectors
def jensen_shannon(vec1, vec2):
    mean1 = np.empty([13, 1])
    mean1 = vec1[0:13]
    #print mean1
    cov1 = np.empty([13,13])
    cov1 = vec1[13:].reshape(13, 13)
    #print cov1
    mean2 = np.empty([13, 1])
    mean2 = vec2[0:13]
    #print mean1
    cov2 = np.empty([13,13])
    cov2 = vec2[13:].reshape(13, 13)
    #print cov1
    mean_m = 0.5 * (mean1 + mean2)
    cov_m = 0.5 * (cov1 + mean1 * np.transpose(mean1)) + 0.5 * (cov2 + mean2 * np.transpose(mean2)) - (mean_m * np.transpose(mean_m))
    div = 0.5 * np.log(np.linalg.det(cov_m)) - 0.25 * np.log(np.linalg.det(cov1)) - 0.25 * np.log(np.linalg.det(cov2))
    #print("JENSEN_SHANNON_DIVERGENCE")    
    if np.isnan(div):
        div = np.inf
        #div = None
    if div <= 0:
        div = div * (-1)
    #print div
    return div

#get 13 mean and 13x13 cov as vectors
def symmetric_kullback_leibler(vec1, vec2):
    mean1 = np.empty([13, 1])
    mean1 = vec1[0:13]
    #print mean1
    cov1 = np.empty([13,13])
    cov1 = vec1[13:].reshape(13, 13)
    #print cov1
    mean2 = np.empty([13, 1])
    mean2 = vec2[0:13]
    #print mean1
    cov2 = np.empty([13,13])
    cov2 = vec2[13:].reshape(13, 13)
    #elem1 = np.trace(cov1 * np.linalg.inv(cov2))
    #elem2 = np.trace(cov2 * np.linalg.inv(cov1))
    #elem3 = np.trace( (np.linalg.inv(cov1) + np.linalg.inv(cov2)) * (mean1 - mean2)**2) 
    d = 13
    div = 0.25 * (np.trace(cov1 * np.linalg.inv(cov2)) + np.trace(cov2 * np.linalg.inv(cov1)) + np.trace( (np.linalg.inv(cov1) + np.linalg.inv(cov2)) * (mean1 - mean2)**2) - 2*d)
    #print div
    return div

song = "music/Electronic/The XX - Intro.mp3"

def naive_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return naive_levenshtein(s2, s1)
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
    #   Pre- Process RP for Euclidean
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
    resultRP = rp_vec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultRP.max(lambda x:x[1])[1]
    min_val = resultRP.min(lambda x:x[1])[1]  
    resultRP = resultRP.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultRP 

def get_neighbors_rh_euclidean(song):
    #########################################################
    #   Pre- Process RH for Euclidean
    #
    rh = sc.textFile("features/out[0-9]*.rh")
    rh = rh.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    rh = rh.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    rh = rh.map(lambda x: x.split(';'))
    rh = rh.map(lambda x: (x[0], x[1].split(",")))
    kv_rh= rh.map(lambda x: (x[0], list(x[1:])))
    rh_vec = kv_rh.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #  
    comparator = rh_vec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultRH = rh_vec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultRH.max(lambda x:x[1])[1]
    min_val = resultRH.min(lambda x:x[1])[1]  
    resultRH = resultRH.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    #resultRH.sortBy(lambda x: x[1]).take(100)    
    return resultRH 

def get_neighbors_bh_euclidean(song):
    #########################################################
    #   Pre- Process BH for Euclidean
    #
    bh = sc.textFile("features/out[0-9]*.bh")
    bh = bh.map(lambda x: x.split(";"))
    kv_bh = bh.map(lambda x: (x[0].replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
    bh_vec = kv_bh.map(lambda x: (x[0], Vectors.dense(x[2]), x[1]))
    #########################################################
    #   Get Neighbors
    #  
    comparator = bh_vec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    #print(np.array(bh_vec.first()[1]))
    #print ( np.array(comparator_value))
    resultBH = bh_vec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultBH.max(lambda x:x[1])[1]
    min_val = resultBH.min(lambda x:x[1])[1]  
    resultBH = resultBH.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    #resultBH.sortBy(lambda x: x[1]).take(100)    
    return resultBH 

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
    resultNotes = notes.map(lambda x: (x[0], naive_levenshtein(str(x[1]), str(comparator_value)), x[1], x[2]))
    max_val = resultNotes.max(lambda x:x[1])[1]
    min_val = resultNotes.min(lambda x:x[1])[1]  
    resultNotes = resultNotes.map(lambda x: (x[0], (float(x[1])-min_val)/(max_val-min_val), x[2], x[3]))  
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
    resultMfcc = mfccVec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultMfcc.max(lambda x:x[1])[1]
    min_val = resultMfcc.min(lambda x:x[1])[1]  
    resultMfcc = resultMfcc.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultMfcc

def get_neighbors_mfcc_skl(song):
    #########################################################
    #   Pre- Process MFCC for SKL and JS
    #
    mfcc = sc.textFile("features/out[0-9]*.mfcckl")            
    mfcc = mfcc.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcc = mfcc.map(lambda x: x.split(';'))
    mfcc = mfcc.map(lambda x: (x[0], x[1].split(',')))
    mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #
    comparator = mfccVec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    resultMfcc = mfccVec.map(lambda x: (x[0], symmetric_kullback_leibler(np.array(x[1]), np.array(comparator_value))))
    resultMfcc = resultMfcc.filter(lambda x: x[1] <= 100)  
    max_val = resultMfcc.max(lambda x:x[1])[1]
    min_val = resultMfcc.min(lambda x:x[1])[1]  
    resultMfcc = resultMfcc.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    #resultMfcc.sortBy(lambda x: x[1]).take(100)
    return resultMfcc

def get_neighbors_mfcc_js(song):
    #########################################################
    #   Pre- Process MFCC for SKL and JS
    #
    mfcc = sc.textFile("features/out[0-9]*.mfcckl")            
    mfcc = mfcc.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcc = mfcc.map(lambda x: x.split(';'))
    mfcc = mfcc.map(lambda x: (x[0], x[1].split(',')))
    mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #
    comparator = mfccVec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    resultMfcc = mfccVec.map(lambda x: (x[0], jensen_shannon(np.array(x[1]), np.array(comparator_value))))
    #drop non valid rows    
    resultMfcc = resultMfcc.filter(lambda x: x[1] != np.inf)    
    max_val = resultMfcc.max(lambda x:x[1])[1]
    min_val = resultMfcc.min(lambda x:x[1])[1]  
    resultMfcc = resultMfcc.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    resultMfcc.sortBy(lambda x: x[1]).take(100)
    return resultMfcc

def get_neighbors_chroma_corr_valid(song):
    #########################################################
    #   Pre- Process Chroma for cross-correlation
    #
    chroma = sc.textFile("features/out[0-9]*.chroma")
    chroma = chroma.map(lambda x: x.split(';'))
    chromaRdd = chroma.map(lambda x: (x[0].replace(' ', '').replace('[', '').replace(']', '').replace(']', ''),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
    chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
    comparator = chromaVec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    #print(np.array(chromaVec.first()[1]))
    #print(np.array(comparator_value))
    resultChroma = chromaVec.map(lambda x: (x[0], chroma_cross_correlate_valid(np.array(x[1]), np.array(comparator_value))))
    #drop non valid rows    
    max_val = resultChroma.max(lambda x:x[1])[1]
    min_val = resultChroma.min(lambda x:x[1])[1]  
    resultChroma = resultChroma.map(lambda x: (x[0], (1 - (x[1]-min_val)/(max_val-min_val))))
    resultChroma.sortBy(lambda x: x[1]).take(100)
    return resultChroma

def get_nearest_neighbors_full(song, outname):
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_rh_euclidean = get_neighbors_rh_euclidean(song)
    neighbors_bh_euclidean = get_neighbors_bh_euclidean(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    neighbors_mfcc_js = get_neighbors_mfcc_js(song)
    neighbors_mfcc_skl = get_neighbors_mfcc_skl(song)
    mergedSim = neighbors_rp_euclidean.join(neighbors_rh_euclidean)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1])))
    mergedSim = mergedSim.join(neighbors_bh_euclidean)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.join(neighbors_chroma)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.join(neighbors_notes)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.join(neighbors_mfcc_eucl)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.join(neighbors_mfcc_skl)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.join(neighbors_mfcc_js)
    mergedSim = mergedSim.map(lambda x: (x[0], list(x[1][0]) + [float(x[1][1])]))
    mergedSim = mergedSim.map(lambda x: (x[0], x[1], float(np.mean(np.array(x[1]))))).sortBy(lambda x: x[2], ascending = True)
    #print mergedSim.first()
    mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

def get_nearest_neighbors_fast(song, outname):
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean)
    mergedSim = mergedSim.join(neighbors_notes)
    #mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    mergedSim = mergedSim.map(lambda x: (x[0], ((x[1][0][1] + x[1][1] + x[1][0][0]) / 3))).sortBy(lambda x: x[1], ascending = True)
    mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Electronic/The XX - Intro.mp3"    #100 testset


result = get_nearest_neighbors_fast(song, "Electro_rdd_fast.csv")
result.sortBy(lambda x: x[1], ascending = True).take(10)

result = get_nearest_neighbors_full(song, "Electro_rdd_full.csv")
result.sortBy(lambda x: x[1], ascending = True).take(10)



