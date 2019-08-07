#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.mllib.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.spatial import distance
#only version 2.1 upwards
#from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
#import org.apache.spark.sql.functions.typedLit
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from scipy.signal import butter, lfilter, freqz, correlate2d, sosfilt
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)

def chroma_cross_correlate_scipy(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([12, length1])
    length2 = chroma2_par.size/12
    chroma2 = np.empty([12, length2])
    if(length1 > length2):
        chroma1 = chroma1_par.reshape(12, length1)
        chroma2 = chroma2_par.reshape(12, length2)
    else:
        chroma2 = chroma1_par.reshape(12, length1)
        chroma1 = chroma2_par.reshape(12, length2)      
    corr = correlate2d(chroma1, chroma2, mode='same')
    #left out according to ellis' 2007 paper
    #transposed_chroma = transposed_chroma / (min(length1, length2))
    index = 5
    mean_line = corr[index]
    #remove offset to get rid of initial filter peak(highpass of jump from 0-20)
    mean_line = mean_line - mean_line[0]
    #print np.max(mean_line)
    sos = butter(1, 0.1, 'high', analog=False, output='sos')
    mean_line = sosfilt(sos, mean_line)[:]
    return np.max(mean_line)

def chroma_cross_correlate_numpy(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([12, length1])
    length2 = chroma2_par.size/12
    chroma2 = np.empty([12, length2])
    if(length1 > length2):
        chroma1 = chroma1_par.reshape(12, length1)
        chroma2 = chroma2_par.reshape(12, length2)
    else:
        chroma2 = chroma1_par.reshape(12, length1)
        chroma1 = chroma2_par.reshape(12, length2)      
    #full
    #correlation = np.zeros([length1 + length2 - 1])
    #valid
    #correlation = np.zeros([max(length1, length2) - min(length1, length2) + 1])
    #same
    correlation = np.zeros([max(length1, length2)])
    for i in range(12):
        correlation = correlation + np.correlate(chroma1[i], chroma2[i], "same")    
    #remove offset to get rid of initial filter peak(highpass of jump from 0-20)
    correlation = correlation - correlation[0]
    sos = butter(1, 0.1, 'high', analog=False, output='sos')
    correlation = sosfilt(sos, correlation)[:]
    return np.max(correlation)

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

#########################################################
#   Pre- Process Chroma for cross-correlation
#

chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))
chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaDf = sqlContext.createDataFrame(chromaRdd, ["id", "chroma"])
chromaVec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))

def get_neighbors_chroma_corr_numpy(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_numpy(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    aggregated = result.agg(F.min(result.distances_corr),F.max(result.distances_corr))
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    return result.withColumn('scaled_corr', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_corr")

def get_neighbors_chroma_corr_scipy(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_scipy(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    aggregated = result.agg(F.min(result.distances_corr),F.max(result.distances_corr))
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    return result.withColumn('scaled_corr', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_corr")

def get_nearest_neighbors_numpy(song, outname):
    neighbors_chroma = get_neighbors_chroma_corr_numpy(song).dropDuplicates()
    mergedSim = neighbors_chroma.orderBy('scaled_corr', ascending=True)
    #mergedSim.limit(20).show()    
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')


def get_nearest_neighbors_scipy(song, outname):
    neighbors_chroma = get_neighbors_chroma_corr_scipy(song).dropDuplicates()
    mergedSim = neighbors_chroma.orderBy('scaled_corr', ascending=True)
    #mergedSim.limit(20).show()    
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Electronic/The XX - Intro.mp3"    #100 testset
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

time_dict = {}

tic1 = int(round(time.time() * 1000))
get_nearest_neighbors_scipy(song, "corr_scipy.csv")
tac1 = int(round(time.time() * 1000))
time_dict['scipy'] = tac1 - tic1

tic2 = int(round(time.time() * 1000))
get_nearest_neighbors_numpy(song, "corr_numpy.csv")
tac2 = int(round(time.time() * 1000))
time_dict['numpy'] = tac2 - tic2

print time_dict


