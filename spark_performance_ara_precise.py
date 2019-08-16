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
#from pyspark.ml.feature import BucketedRandomProjectionLSH
#from pyspark.mllib.linalg import Vectors
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
import time
from scipy.signal import butter, lfilter, freqz, correlate2d, sosfilt

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
#from pyspark.sql import SparkSession

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

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
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
    correlation = np.zeros([max(length1, length2)])
    for i in range(12):
        correlation = correlation + np.correlate(chroma1[i], chroma2[i], "same")    
    correlation = correlation - correlation[0]
    sos = butter(1, 0.1, 'high', analog=False, output='sos')
    correlation = sosfilt(sos, correlation)[:]
    return np.max(correlation)


#get 13 mean and 13x13 cov as vectors
def jensen_shannon(vec1, vec2):
    mean1 = np.empty([13, 1])
    mean1 = vec1[0:13]
    cov1 = np.empty([13,13])
    cov1 = vec1[13:].reshape(13, 13)
    mean2 = np.empty([13, 1])
    mean2 = vec2[0:13]
    cov2 = np.empty([13,13])
    cov2 = vec2[13:].reshape(13, 13)
    mean_m = 0.5 * (mean1 + mean2)
    cov_m = 0.5 * (cov1 + mean1 * np.transpose(mean1)) + 0.5 * (cov2 + mean2 * np.transpose(mean2)) - (mean_m * np.transpose(mean_m))
    div = 0.5 * np.log(np.linalg.det(cov_m)) - 0.25 * np.log(np.linalg.det(cov1)) - 0.25 * np.log(np.linalg.det(cov2))
    #print("JENSEN_SHANNON_DIVERGENCE")    
    if np.isnan(div):
        div = np.inf
    if div <= 0:
        div = div * (-1)
    return div

def get_neighbors_chroma_corr_valid_df(song):
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
    df_vec = chromaDf
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    aggregated = result.agg(F.min(result.distances_corr),F.max(result.distances_corr))
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    return result.withColumn('scaled_corr', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_corr")

def get_neighbors_mfcc_js_df(song):
    #########################################################
    #   Pre- Process MFCC for SKL and JS
    #
    mfcc = sc.textFile("features[0-9]*/out[0-9]*.mfcckl")            
    mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
    mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcc = mfcc.map(lambda x: x.split(';'))
    mfcc = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
    mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
    mfccDfMerged = sqlContext.createDataFrame(mfccVec, ["id", "features"])
    df_vec = mfccDfMerged
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_js', distance_udf(F.col('features'))).select("id", "distances_js")
    #drop non valid rows    
    #result = result.filter(result.distances_js.isNotNull())
    result = result.filter(result.distances_js != np.inf)    
    aggregated = result.agg(F.min(result.distances_js),F.max(result.distances_js))
    max_val = aggregated.collect()[0]["max(distances_js)"]
    min_val = aggregated.collect()[0]["min(distances_js)"]
    return result.withColumn('scaled_js', (result.distances_js-min_val)/(max_val-min_val)).select("id", "scaled_js")

def get_neighbors_rp_euclidean_df(song):
    #########################################################
    #   Pre- Process RH and RP for Euclidean
    #
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.split(","))
    kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = sqlContext.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_rp', distance_udf(F.col('features'))).select("id", "distances_rp")
    aggregated = result.agg(F.min(result.distances_rp),F.max(result.distances_rp))
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")

def get_neighbors_rp_euclidean_rdd(song):
    #########################################################
    #   Pre- Process RP for Euclidean
    #
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.replace(' ', '').replace(';', ','))
    rp = rp.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    rp = rp.map(lambda x: x.split(';'))
    rp = rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(",")))
    kv_rp= rp.map(lambda x: (x[0], list(x[1:])))
    rp_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #  
    comparator = rp_vec.lookup(song.replace(' ', '').replace(';', ','))
    comparator_value = comparator[0]
    resultRP = rp_vec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultRP.max(lambda x:x[1])[1]
    min_val = resultRP.min(lambda x:x[1])[1]  
    resultRP = resultRP.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultRP 

def get_neighbors_mfcc_js_rdd(song):
    #########################################################
    #   Pre- Process MFCC for SKL and JS
    #
    mfcc = sc.textFile("features[0-9]*/out[0-9]*.mfcckl")            
    mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
    mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcc = mfcc.map(lambda x: x.split(';'))
    mfcc = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
    mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #
    comparator = mfccVec.lookup(song.replace(' ', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    resultMfcc = mfccVec.map(lambda x: (x[0], jensen_shannon(np.array(x[1]), np.array(comparator_value))))
    #drop non valid rows    
    resultMfcc = resultMfcc.filter(lambda x: x[1] != np.inf)    
    max_val = resultMfcc.max(lambda x:x[1])[1]
    min_val = resultMfcc.min(lambda x:x[1])[1]  
    resultMfcc = resultMfcc.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    resultMfcc.sortBy(lambda x: x[1]).take(100)
    return resultMfcc

def get_neighbors_chroma_corr_valid_rdd(song):
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
    comparator = chromaVec.lookup(song.replace(' ', '').replace(';', ','))
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


def get_neighbors_mfcc_js_merge(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl'))).select("id", "distances_js")
    result = result.filter(result.distances_js != np.inf)    
    return result

def get_neighbors_rp_euclidean_merge(song, featureDF):
    comparator_value = song[0]["rp"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")
    return result

def get_neighbors_chroma_corr_valid_merge(song, featureDF):
    comparator_value = song[0]["chroma"]
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    return result

def perform_scaling(unscaled_df):
    aggregated = unscaled_df.agg(
        F.min(unscaled_df.distances_rp),F.max(unscaled_df.distances_rp),F.mean(unscaled_df.distances_rp),F.stddev(unscaled_df.distances_rp),
        F.min(unscaled_df.distances_corr),F.max(unscaled_df.distances_corr),F.mean(unscaled_df.distances_corr),F.stddev(unscaled_df.distances_corr),
        F.min(unscaled_df.distances_js),F.max(unscaled_df.distances_js),F.mean(unscaled_df.distances_js),F.stddev(unscaled_df.distances_js))
    ##############################
    #var_val = aggregated.collect()[0]["stddev_samp(distances_bh)"]
    #mean_val = aggregated.collect()[0]["avg(distances_bh)"]
    ##############################
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    result = unscaled_df.withColumn('scaled_rp', (unscaled_df.distances_rp-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    result = result.withColumn('scaled_chroma', (1 - (unscaled_df.distances_corr-min_val)/(max_val-min_val)))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_js)"]
    min_val = aggregated.collect()[0]["min(distances_js)"]
    result = result.withColumn('scaled_js', (unscaled_df.distances_js-min_val)/(max_val-min_val)).select("id", "scaled_rp", "scaled_chroma", "scaled_js")
    ##############################
    return result

def get_nearest_neighbors_speed(song, outname):
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
    #########################################################
    #   Gather all features in one dataframe
    #
    featureDF = chromaDf.join(mfccDfMerged, on=["id"], how='inner')
    featureDF = featureDF.join(rp_df, on=['id'], how='inner').dropDuplicates().persist()
    #########################################################
    #  16 Nodes, 192GB RAM each, 36 cores each (+ hyperthreading = 72)
    #   -> max 1152 executors
    fullFeatureDF = featureDF.repartition(repartition_count)
    #fullFeatureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')
    song = fullFeatureDF.filter(featureDF.id == song).collect()
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_merge(song, fullFeatureDF).persist()
    neighbors_mfcc_js = get_neighbors_mfcc_js_merge(song, fullFeatureDF).persist()
    neighbors_chroma = get_neighbors_chroma_corr_valid_merge(song, fullFeatureDF).persist()
    mergedSim = neighbors_mfcc_js.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner').dropDuplicates().persist()
    neighbors_rp_euclidean.unpersist()
    neighbors_mfcc_js.unpersist()
    neighbors_chroma.unpersist()
    scaledSim = perform_scaling(mergedSim).persist()
    mergedSim.unpersist()
    scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_chroma + scaledSim.scaled_rp + scaledSim.scaled_js) / 3)
    scaledSim = scaledSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    scaledSim.show()
    #scaledSim.toPandas().to_csv(outname, encoding='utf-8')
    scaledSim.unpersist()
    return scaledSim

def get_nearest_neighbors_dataframe(song, outname):
    neighbors_mfcc_js = get_neighbors_mfcc_js_df(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_df(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid_df(song)
    mergedSim = neighbors_mfcc_js.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_corr + mergedSim.scaled_rp + mergedSim.scaled_js) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    #mergedSim.toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

def get_nearest_neighbors_rdd(song, outname):
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_rdd(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid_rdd(song)
    neighbors_mfcc_js = get_neighbors_mfcc_js_rdd(song)
    mergedSim = neighbors_mfcc_js.join(neighbors_rp_euclidean)
    mergedSim = mergedSim.join(neighbors_chroma)
    mergedSim = mergedSim.map(lambda x: (x[0], ((x[1][0][1] + x[1][1] + x[1][0][0]) / 3))).sortBy(lambda x: x[1], ascending = True)
    #mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

song = "music/Let_It_Be/beatles+Let_It_Be+06-Let_It_Be.mp3"
#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3"
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

tic2 = int(round(time.time() * 1000))
res = get_nearest_neighbors_speed(song, "MERGED.csv")
tac2 = int(round(time.time() * 1000))
time_dict['TOTAL MERGED']= tac2 - tic2
res.toPandas().to_csv("MERGED.csv", encoding='utf-8') 

tic4 = int(round(time.time() * 1000))
res =get_nearest_neighbors_dataframe(song, "DF.csv")
tac4 = int(round(time.time() * 1000))
time_dict['TOTAL DF']= tac4 - tic4
res.toPandas().to_csv("DF.csv", encoding='utf-8') 

tic5 = int(round(time.time() * 1000))
res = get_nearest_neighbors_rdd(song, "RDD.csv")
tac5 = int(round(time.time() * 1000))
time_dict['TOTAL RDD']= tac5 - tic5
res.map(lambda x: (x[0], float(x[1]))).toDF().toPandas().to_csv("RDD.csv", encoding='utf-8') 

print time_dict


