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
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d, sosfilt
import time
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

total1 = int(round(time.time() * 1000))

confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confCluster.set("spark.driver.memory", "64g")
confCluster.set("spark.executor.memory", "64g")
confCluster.set("spark.driver.memoryOverhead", "32g")
confCluster.set("spark.executor.memoryOverhead", "32g")
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb
#confCluster.set("yarn.nodemanager.resource.memory-mb", "196608")
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb
confCluster.set("spark.yarn.executor.memoryOverhead", "4096")
#set cores of each executor and the driver -> less than avail -> more executors spawn
confCluster.set("spark.driver.cores", "36")
confCluster.set("spark.executor.cores", "36")
confCluster.set("spark.dynamicAllocation.enabled", "True")
confCluster.set("spark.dynamicAllocation.minExecutors", "16")
confCluster.set("spark.dynamicAllocation.maxExecutors", "32")
confCluster.set("yarn.nodemanager.vmem-check-enabled", "false")
repartition_count = 32


sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)
time_dict = {}

def chroma_cross_correlate_valid(chroma1_par, chroma2_par):
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

def preprocess_features():
    chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma", minPartitions=repartition_count)
    chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
    chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    chroma = chroma.map(lambda x: x.split(';'))
    #try to filter out empty elements
    chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
    chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
    chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
    chromaDf = sqlContext.createDataFrame(chromaVec, ["id", "chroma"]).repartition(repartition_count)
    return chromaDf

def get_distances(song, chromaDf):
    df_vec = chromaDf
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    aggregated = result.agg(F.min(result.distances_corr),F.max(result.distances_corr))
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    chromaDf.unpersist()
    return result.withColumn('scaled_dist', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_dist")

if len (sys.argv) < 2:
    songname = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3" #1517 artists
    song2 = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3" #1517 artists
    songname = songname.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
    song2 = song2.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
else: 
    songname = sys.argv[1]
    song2 = sys.argv[1]
    songname = songname.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
    song2 = song2.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')


featureDf = preprocess_features()#.persist()
#print(featureDf.first())

tic1 = int(round(time.time() * 1000))
neighbors = get_distances(songname, featureDf)
neighbors = neighbors.orderBy('scaled_dist', ascending=True)#.persist()
neighbors.show()
neighbors.toPandas().to_csv("neighbors.csv", encoding='utf-8')
#neighbors.unpersist()
tac1 = int(round(time.time() * 1000))
time_dict['time: ']= tac1 - tic1

print time_dict
