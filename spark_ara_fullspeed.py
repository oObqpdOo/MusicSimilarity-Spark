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

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confCluster.set("spark.driver.memory", "3g")
confCluster.set("spark.executor.memory", "3g")
confCluster.set("spark.driver.memoryOverhead", "3g")
confCluster.set("spark.executor.memoryOverhead", "3g")
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb
confCluster.set("yarn.nodemanager.resource.memory-mb", "8192")
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb
confCluster.set("spark.yarn.executor.memoryOverhead", "2048")
#set cores of each executor and the driver -> less than avail -> more executors spawn
confCluster.set("spark.driver.cores", "4")
confCluster.set("spark.executor.cores", "4")
#
confCluster.set("spark.dynamicAllocation.enabled", "True")
confCluster.set("yarn.nodemanager.vmem-check-enabled", "false")

sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)

def chroma_cross_correlate(chroma1_par, chroma2_par):
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
    corr = sp.signal.correlate2d(chroma1, chroma2, mode='same') 
    #print np.max(mean_line)
    return np.max(corr)

def chroma_cross_correlate_full(chroma1_par, chroma2_par):
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
    corr = sp.signal.correlate2d(chroma1, chroma2, mode='full')
    transposed_chroma = corr.transpose()  
    #print "length1: " + str(length1)
    #print "length2: " + str(length2)
    #transposed_chroma = transposed_chroma / (min(length1, length2))
    index = np.where(transposed_chroma == np.amax(transposed_chroma))
    index = int(index[0])
    #print "index: " + str(index)
    transposed_chroma = transposed_chroma.transpose()
    transposed_chroma = np.transpose(transposed_chroma)
    mean_line = transposed_chroma[index]
    sos = sp.signal.butter(1, 0.1, 'high', analog=False, output='sos')
    mean_line = sp.signal.sosfilt(sos, mean_line)
    #print np.max(mean_line)
    return np.max(mean_line)


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


list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())


#########################################################
#   Pre- Process RH and RP for Euclidean
#

rh = sc.textFile("features[0-9]*/out[0-9]*.rh")
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))

rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))

#########################################################
#   Pre- Process BH for Euclidean
#

bh = sc.textFile("features[0-9]*/out[0-9]*.bh")
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))

#########################################################
#   Pre- Process Notes for Levenshtein
#

notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))

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
#   Pre- Process MFCC for Euclidean
#

mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccEucDfMerged = sqlContext.createDataFrame(mfccVec, ["id", "features"])

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

def get_neighbors_chroma_corr_valid(song):
    df_vec = chromaDf
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    aggregated = result.agg(F.min(result.distances_corr),F.max(result.distances_corr))
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    return result.withColumn('scaled_corr', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_corr")

def get_neighbors_mfcc_euclidean(song):
    df_vec = mfccEucDfMerged
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_mfcc', distance_udf(F.col('features'))).select("id", "distances_mfcc")
    aggregated = result.agg(F.min(result.distances_mfcc),F.max(result.distances_mfcc))
    max_val = aggregated.collect()[0]["max(distances_mfcc)"]
    min_val = aggregated.collect()[0]["min(distances_mfcc)"]
    return result.withColumn('scaled_mfcc', (result.distances_mfcc-min_val)/(max_val-min_val)).select("id", "scaled_mfcc")

def get_neighbors_mfcc_skl(song):
    df_vec = mfccDfMerged
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_skl', distance_udf(F.col('features'))).select("id", "distances_skl")
    #thresholding 
    #result = result.filter(result.distances_skl <= 1000)  
    aggregated = result.agg(F.min(result.distances_skl),F.max(result.distances_skl))
    max_val = aggregated.collect()[0]["max(distances_skl)"]
    min_val = aggregated.collect()[0]["min(distances_skl)"]
    return result.withColumn('scaled_skl', (result.distances_skl-min_val)/(max_val-min_val)).select("id", "scaled_skl")

def get_neighbors_mfcc_js(song):
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

def get_neighbors_rp_euclidean(song):
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

def get_neighbors_rh_euclidean(song):
    comparator = kv_rh.lookup(song)
    comparator_value = comparator[0]
    df = sqlContext.createDataFrame(kv_rh, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_rh', distance_udf(F.col('features'))).select("id", "distances_rh")
    aggregated = result.agg(F.min(result.distances_rh),F.max(result.distances_rh))
    max_val = aggregated.collect()[0]["max(distances_rh)"]
    min_val = aggregated.collect()[0]["min(distances_rh)"]
    return result.withColumn('scaled_rh', (result.distances_rh-min_val)/(max_val-min_val)).select("id", "scaled_rh")

def get_neighbors_notes(song):
    df = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])
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

def get_neighbors_bh_euclidean(song):
    df = sqlContext.createDataFrame(kv_bh, ["id", "bpm", "features"])
    filterDF = df.filter(df.id == song)
    comparator_value = filterDF.collect()[0][2]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df.withColumn('distances_bh', distance_udf(F.col('features'))).select("id", "bpm", "distances_bh")
    aggregated = result.agg(F.min(result.distances_bh),F.max(result.distances_bh))
    max_val = aggregated.collect()[0]["max(distances_bh)"]
    min_val = aggregated.collect()[0]["min(distances_bh)"]
    return result.withColumn('scaled_bh', (result.distances_bh-min_val)/(max_val-min_val)).select("id", "bpm", "scaled_bh")

def get_nearest_neighbors_full(song, outname):
    neighbors_mfcc_skl = get_neighbors_mfcc_skl(song)
    neighbors_mfcc_js = get_neighbors_mfcc_js(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_rh_euclidean = get_neighbors_rh_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    neighbors_bh_euclidean = get_neighbors_bh_euclidean(song)
    #print neighbors_mfcc_skl.first()
    #print neighbors_rp_euclidean.first()
    #neighbors_notes.show()
    #JOIN could also left_inner and handle 'nones'
    mergedSim = neighbors_mfcc_skl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_rh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_eucl, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_js, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_bh + mergedSim.scaled_mfcc + mergedSim.scaled_corr + mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_skl + mergedSim.scaled_js + mergedSim.scaled_rh) / 8)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    #mergedSim.show()
    out_name = outname#"output.csv"
    mergedSim.toPandas().to_csv(out_name, encoding='utf-8')


def get_nearest_neighbors_fast(song, outname):
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    neighbors_rh_euclidean = get_neighbors_rh_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rh + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')


def get_nearest_neighbors_precise(song, outname):
    neighbors_mfcc_js = get_neighbors_mfcc_js(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    mergedSim = neighbors_mfcc_js.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_corr + mergedSim.scaled_rp + mergedSim.scaled_js) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
def get_nearest_neighbors_test(song, outname):
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_bh_euclidean = get_neighbors_bh_euclidean(song)
    neighbors_rh_euclidean = get_neighbors_rh_euclidean(song)    
    mergedSim = neighbors_notes.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_eucl, on=['id'], how='inner')
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_bh + mergedSim.scaled_corr + mergedSim.scaled_levenshtein + mergedSim.scaled_mfcc + mergedSim.scaled_rp + mergedSim.scaled_rh) / 6)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')

song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3"

song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

#get_nearest_neighbors_fast(song, "result_df_fast.csv")
#get_nearest_neighbors_precise(song, "result_df_precise.csv")
get_nearest_neighbors_full(song, "result_df_full.csv")

#get_nearest_neighbors_test(song, "result_df_test.csv")
#get_nearest_neighbors_precise(song, "result_df_precise.csv")






