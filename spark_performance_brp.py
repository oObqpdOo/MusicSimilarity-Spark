#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.ml.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.spatial import distance
from pyspark.ml.feature import BucketedRandomProjectionLSH
#from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.ml.linalg import Vectors, VectorUDT
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
from scipy.signal import butter, lfilter, freqz, correlate2d

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confCluster.set("spark.driver.memory", "1g")
confCluster.set("spark.executor.memory", "1g")
confCluster.set("spark.driver.memoryOverhead", "500m")
confCluster.set("spark.executor.memoryOverhead", "500m")
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb
#confCluster.set("yarn.nodemanager.resource.memory-mb", "192000")
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb
confCluster.set("spark.yarn.executor.memoryOverhead", "512")
#set cores of each executor and the driver -> less than avail -> more executors spawn
confCluster.set("spark.driver.cores", "1")
confCluster.set("spark.executor.cores", "1")
confCluster.set("spark.dynamicAllocation.enabled", "True")
confCluster.set("spark.dynamicAllocation.minExecutors", "4")
confCluster.set("spark.dynamicAllocation.maxExecutors", "4")
confCluster.set("yarn.nodemanager.vmem-check-enabled", "false")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("cluster").appName("MusicSimilarity").getOrCreate()

def get_neighbors_rp_euclidean_dataframe(song):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.split(","))
    kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    comparator_value = Vectors.dense(comparator[0])
    df = spark.createDataFrame(kv_rp, ["id", "features"]).persist()
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_rp', distance_udf(F.col('features'))).select("id", "distances_rp")
    aggregated = result.agg(F.min(result.distances_rp),F.max(result.distances_rp))
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")


def get_neighbors_mfcc_euclidean_dataframe(song):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    #########################################################
    #   Pre- Process MFCC for Euclidean
    #

    mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
    mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace(';', ','))
    mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcceuc = mfcceuc.map(lambda x: x.split(';'))
    mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
    mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
    mfccEucDfMerged = spark.createDataFrame(mfccVec, ["id", "features"])
    df_vec = mfccEucDfMerged
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_mfcc', distance_udf(F.col('features'))).select("id", "distances_mfcc")
    aggregated = result.agg(F.min(result.distances_mfcc),F.max(result.distances_mfcc))
    max_val = aggregated.collect()[0]["max(distances_mfcc)"]
    min_val = aggregated.collect()[0]["min(distances_mfcc)"]
    return result.withColumn('scaled_mfcc', (result.distances_mfcc-min_val)/(max_val-min_val)).select("id", "scaled_mfcc")


def get_neighbors_rp_euclidean_dataframe_brp(song):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.split(","))
    kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=100.0)
    model = brp.fit(df_vec)
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    #model.approxSimilarityJoin(df_vec, comparator_value, 3.0, distCol="EuclideanDistance")
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol")
    aggregated = result.agg(F.min(result.distCol),F.max(result.distCol))
    max_val = aggregated.collect()[0]["max(distCol)"]
    min_val = aggregated.collect()[0]["min(distCol)"]
    return result.withColumn('scaled_rp', (result.distCol-min_val)/(max_val-min_val)).select("id", "scaled_rp")


def get_neighbors_mfcc_euclidean_dataframe_brp(song):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    #########################################################
    #   Pre- Process MFCC for Euclidean
    #

    mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
    mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace(';', ','))
    mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcceuc = mfcceuc.map(lambda x: x.split(';'))
    mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
    mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
    mfccEucDfMerged = spark.createDataFrame(mfccVec, ["id", "features"])
    df_vec = mfccEucDfMerged.persist()
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=100.0)
    model = brp.fit(df_vec)
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol")
    aggregated = result.agg(F.min(result.distCol),F.max(result.distCol))
    max_val = aggregated.collect()[0]["max(distCol)"]
    min_val = aggregated.collect()[0]["min(distCol)"]
    return result.withColumn('scaled_mfcc', (result.distCol-min_val)/(max_val-min_val)).select("id", "scaled_mfcc")


def get_neighbors_notes_dataframe(song):
    #########################################################
    #   Pre- Process Notes for Levenshtein
    #
    notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
    notes = notes.map(lambda x: x.split(';'))
    notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
    notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
    df = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])
    filterDF = df.filter(df.id == song)
    comparator_value = filterDF.collect()[0][3] 
    df_merged = df.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    #df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
    result = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
    aggregated = result.agg(F.min(result.distances_levenshtein),F.max(result.distances_levenshtein))
    max_val = aggregated.collect()[0]["max(distances_levenshtein)"]
    min_val = aggregated.collect()[0]["min(distances_levenshtein)"]
    return result.withColumn('scaled_levenshtein', (result.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "key", "scale", "scaled_levenshtein")

def get_nearest_neighbors_dataframe(song, outname):
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean_dataframe(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_dataframe(song)
    neighbors_notes = get_neighbors_notes_dataframe(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.limit(20).show()    
    #mergedSim.toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

def get_nearest_neighbors_brp_df(song, outname):
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean_dataframe_brp(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_dataframe_brp(song)
    neighbors_notes = get_neighbors_notes_dataframe(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.limit(20).show()    
    #mergedSim.toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
#song = "music/Electronic/The XX - Intro.mp3"    #100 testset
song = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3"
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

time_dict = {}

tic1 = int(round(time.time() * 1000))
ret = get_nearest_neighbors_brp_df(song, "perf_dataframe_brp.csv")
tac1 = int(round(time.time() * 1000))
time_dict['dataframe_brp']= tac1 - tic1

ret.toPandas().to_csv("brp.csv", encoding='utf-8')

tic2 = int(round(time.time() * 1000))
ret = get_nearest_neighbors_dataframe(song, "perf_dataframe_speed.csv")
tac2 = int(round(time.time() * 1000))
time_dict['dataframe_speed']= tac2 - tic2

ret.toPandas().to_csv("euc.csv", encoding='utf-8')

print time_dict




