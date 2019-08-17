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
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import sys

total1 = int(round(time.time() * 1000))

confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confCluster.set("spark.driver.memory", "64g")
confCluster.set("spark.executor.memory", "64g")
confCluster.set("spark.driver.memoryOverhead", "32g")
confCluster.set("spark.executor.memoryOverhead", "32g")
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb
#confCluster.set("yarn.nodemanager.resource.memory-mb", "192000")
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb
confCluster.set("spark.yarn.executor.memoryOverhead", "4096")
#set cores of each executor and the driver -> less than avail -> more executors spawn
confCluster.set("spark.driver.cores", "32")
confCluster.set("spark.executor.cores", "32")
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

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

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
    if (is_invertible(cov1) and is_invertible(cov2)):
        d = 13
        div = 0.25 * (np.trace(cov1 * np.linalg.inv(cov2)) + np.trace(cov2 * np.linalg.inv(cov1)) + np.trace( (np.linalg.inv(cov1) + np.linalg.inv(cov2)) * (mean1 - mean2)**2) - 2*d)
    else: 
        div = np.inf
        print("ERROR: NON INVERTIBLE SINGULAR COVARIANCE MATRIX \n\n\n")    
    #print div
    return div

tic1 = int(round(time.time() * 1000))
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

#########################################################
#   Pre- Process RH and RP for Euclidean
#
rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rp_df = sqlContext.createDataFrame(kv_rp, ["id", "rp"])
rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))


#########################################################
#   Pre- Process Chroma for cross-correlation
#

chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
chroma = chroma.map(lambda x: x.split(';'))
#try to filter out empty elements
chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
chromaDf = sqlContext.createDataFrame(chromaVec, ["id", "chroma"])

#########################################################
#   Pre- Process MFCC for SKL and JS
#

mfcc = sc.textFile("features[0-9]*/out[0-9]*.mfcckl")            
mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccDfMerged = sqlContext.createDataFrame(mfccVec, ["id", "mfccSkl"])

#print(rh_df.count())
#print(rp_df.count())
#print(bh_df.count())
#print(notesDf.count())
#print(chromaDf.count())
#print(mfccDfMerged.count())
#print(mfccEucDfMerged.count())

#########################################################
#   Gather all features in one dataframe
#
featureDF = chromaDf.join(mfccDfMerged, on=["id"], how='inner')
featureDF = featureDF.join(rp_df, on=['id'], how='inner').dropDuplicates().persist()

#Force lazy evaluation to evaluate with an action
#trans = featureDF.count()
#print(featureDF.count())


#########################################################
#  16 Nodes, 192GB RAM each, 36 cores each (+ hyperthreading = 72)
#   -> max 1152 executors

fullFeatureDF = featureDF.repartition(repartition_count).persist()
#print(fullFeatureDF.count())
#fullFeatureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')
tac1 = int(round(time.time() * 1000))
time_dict['PREPROCESS: ']= tac1 - tic1

def get_neighbors_mfcc_js(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl'))).select("id", "distances_js")
    unscaled_df = result.filter(result.distances_js != np.inf)   
    ##############################
    aggregated = unscaled_df.agg(F.min(unscaled_df.distances_js),F.max(unscaled_df.distances_js),F.mean(unscaled_df.distances_js))
    mean_val = aggregated.collect()[0]["avg(distances_js)"]
    max_val = aggregated.collect()[0]["max(distances_js)"]
    min_val = aggregated.collect()[0]["min(distances_js)"]
    result = unscaled_df.filter(unscaled_df.distances_js < mean_val)
    result = result.withColumn('scaled_js', (unscaled_df.distances_js-min_val)/(max_val-min_val)).select("id", "scaled_js")   
    return result

def get_neighbors_rp_euclidean(song, featureDF):
    comparator_value = song[0]["rp"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    unscaled_df = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")
    ##############################
    aggregated = unscaled_df.agg(F.min(unscaled_df.distances_rp),F.max(unscaled_df.distances_rp),F.mean(unscaled_df.distances_rp))
    mean_val = aggregated.collect()[0]["avg(distances_rp)"]
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    result = unscaled_df.filter(unscaled_df.distances_rp < mean_val) 
    result = result.withColumn('scaled_rp', (unscaled_df.distances_rp-min_val)/(max_val-min_val)).select("id","scaled_rp") 
    return result

def get_neighbors_chroma_corr_valid(song, featureDF):
    comparator_value = song[0]["chroma"]
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    unscaled_df = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    ##############################
    aggregated = unscaled_df.agg(F.min(unscaled_df.distances_corr),F.max(unscaled_df.distances_corr),F.mean(unscaled_df.distances_corr))
    mean_val = aggregated.collect()[0]["avg(distances_corr)"]
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    #!!CAREFUL -> CHROMA NOT SMALLER, BUT GREATER THAN MEAN_VAL
    result = unscaled_df.filter(unscaled_df.distances_corr > mean_val)   
    result = result.withColumn('scaled_chroma', (1 - (unscaled_df.distances_corr-min_val)/(max_val-min_val))).select("id","scaled_chroma")
    return result

def get_nearest_neighbors_filter(song, outname, fullFeatureDF):
    tic1 = int(round(time.time() * 1000))
    song = fullFeatureDF.filter(featureDF.id == song).collect()#
    tac1 = int(round(time.time() * 1000))
    time_dict['COMP: ']= tac1 - tic1 

    tic1 = int(round(time.time() * 1000))
    neighbors_chroma = get_neighbors_chroma_corr_valid(song, fullFeatureDF).persist()
    fullFeatureDF = fullFeatureDF.filter(fullFeatureDF.id == neighbors_chroma.id)
    tac1 = int(round(time.time() * 1000))
    time_dict['CHROMA: ']= tac1 - tic1

    tic1 = int(round(time.time() * 1000))
    neighbors_mfcc_js = get_neighbors_mfcc_js(song, fullFeatureDF).persist()
    fullFeatureDF = fullFeatureDF.filter(fullFeatureDF.id == neighbors_mfcc_js.id)
    tac1 = int(round(time.time() * 1000))
    time_dict['JS: ']= tac1 - tic1

    tic1 = int(round(time.time() * 1000))
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song, fullFeatureDF).persist()
    fullFeatureDF = fullFeatureDF.filter(fullFeatureDF.id == neighbors_rp_euclidean.id)
    tac1 = int(round(time.time() * 1000))
    time_dict['RP: ']= tac1 - tic1

    tic1 = int(round(time.time() * 1000))
    mergedSim = neighbors_chroma.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_js, on=['id'], how='inner').dropDuplicates().persist()
    tac1 = int(round(time.time() * 1000))
    time_dict['JOIN: ']= tac1 - tic1

    tic1 = int(round(time.time() * 1000))
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_chroma + mergedSim.scaled_rp + mergedSim.scaled_js) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    mergedSim.show()
    #scaledSim.toPandas().to_csv(outname, encoding='utf-8')

    mergedSim.unpersist()
    neighbors_rp_euclidean.unpersist()
    neighbors_mfcc_js.unpersist()
    neighbors_chroma.unpersist()

    tac1 = int(round(time.time() * 1000))
    time_dict['AGG_F: ']= tac1 - tic1
    return mergedSim

if len (sys.argv) < 2:
    #song = "music/Electronic/The XX - Intro.mp3"    #100 testset
    song = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3"
else: 
    song = sys.argv[1]
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

tic1 = int(round(time.time() * 1000))
res = get_nearest_neighbors_filter(song, "FILTER_REFINE.csv", fullFeatureDF).persist()
tac1 = int(round(time.time() * 1000))
time_dict['FILTER_FULL: ']= tac1 - tic1

total2 = int(round(time.time() * 1000))
time_dict['FILTER_TOTAL: ']= total2 - total1

tic2 = int(round(time.time() * 1000))
res.toPandas().to_csv("FILTER_REFINE.csv", encoding='utf-8')
tac2 = int(round(time.time() * 1000))
time_dict['CSV_F: ']= tac2 - tic2

print time_dict

featureDF.unpersist()


