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
from scipy.signal import butter, lfilter, freqz, correlate2d, sosfilt
import sys
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("cluster").appName("MusicSimilarity").getOrCreate()

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
#song = "music/HURRICANE1.mp3"              #small testset


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

#########################################################
#   List to Vector UDF
#

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
repartition_count = 40

#########################################################
#   Pre- Process RH and RP for Euclidean
#
rp = sc.textFile("features[0-9]*/out[0-9]*.rp").coalesce(repartition_count)
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rp_df = spark.createDataFrame(kv_rp, ["id", "rp"])
rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))
rh = sc.textFile("features[0-9]*/out[0-9]*.rh").coalesce(repartition_count)
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rh_df = spark.createDataFrame(kv_rh, ["id", "rh"])
rh_df = rh_df.select(rh_df["id"],list_to_vector_udf(rh_df["rh"]).alias("rh"))

#########################################################
#   Pre- Process BH for Euclidean
#
bh = sc.textFile("features[0-9]*/out[0-9]*.bh").coalesce(repartition_count)
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
bh_df = spark.createDataFrame(kv_bh, ["id", "bpm", "bh"])

#########################################################
#   Pre- Process Notes for Levenshtein
#
notes = sc.textFile("features[0-9]*/out[0-9]*.notes").coalesce(repartition_count)
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
notesDf = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])

#########################################################
#   Pre- Process Chroma for cross-correlation
#

chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma").coalesce(repartition_count)
chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
chroma = chroma.map(lambda x: x.split(';'))
#try to filter out empty elements
chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
chromaDf = spark.createDataFrame(chromaVec, ["id", "chroma"])

#########################################################
#   Pre- Process MFCC for Euclidean
#

mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc").coalesce(repartition_count)
mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccEucDfMerged = spark.createDataFrame(mfccVec, ["id", "mfccEuc"])

#########################################################
#   Pre- Process MFCC for SKL and JS
#

mfcc = sc.textFile("features[0-9]*/out[0-9]*.mfcckl")   .coalesce(repartition_count)         
mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccDfMerged = spark.createDataFrame(mfccVec, ["id", "mfccSkl"])


#########################################################
#   Gather all features in one dataframe
#
featureDF = mfccEucDfMerged.join(mfccDfMerged, on=["id"], how='inner')
featureDF = featureDF.join(chromaDf, on=['id'], how='inner')
featureDF = featureDF.join(notesDf, on=['id'], how='inner')
featureDF = featureDF.join(rp_df, on=['id'], how='inner')
featureDF = featureDF.join(rh_df, on=['id'], how='inner')
#orig_featureDF IS THE ORIGINAL
orig_featureDF = featureDF.join(bh_df, on=['id'], how='inner').dropDuplicates().coalesce(repartition_count)#.persist()

def get_neighbors_mfcc_skl(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    featureDF = featureDF.withColumn('distances_skl', distance_udf(F.col('mfccSkl')))
    #thresholding 
    #featureDF = featureDF.filter(featureDF.distances_skl <= 100)  
    featureDF = featureDF.filter(featureDF.distances_skl != np.inf)
    return featureDF

def get_neighbors_mfcc_js(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    featureDF = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl')))
    featureDF = featureDF.filter(featureDF.distances_js != np.inf)    
    return featureDF

def get_neighbors_rp_euclidean(song, featureDF):
    comparator_value = song[0]["rp"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    featureDF = featureDF.withColumn('distances_rp', distance_udf(F.col('rp')))
    return featureDF

def get_neighbors_rh_euclidean(song, featureDF):
    comparator_value = song[0]["rh"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    featureDF = featureDF.withColumn('distances_rh', distance_udf(F.col('rh')))
    return featureDF

def get_neighbors_bh_euclidean(song, featureDF):
    comparator_value = song[0]["bh"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    featureDF = featureDF.withColumn('distances_bh', distance_udf(F.col('bh')))
    return featureDF

def get_neighbors_mfcc_euclidean(song, featureDF):
    comparator_value = song[0]["mfccEuc"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    featureDF = featureDF.withColumn('distances_mfcc', distance_udf(F.col('mfccEuc')))
    return featureDF

def get_neighbors_notes(song, featureDF):
    comparator_value = song[0]["notes"]
    featureDF = featureDF.withColumn("compare", lit(comparator_value))
    featureDF = featureDF.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    return featureDF

def get_neighbors_chroma_corr_valid(song, featureDF):
    comparator_value = song[0]["chroma"]
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    featureDF = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma')))
    return featureDF

def perform_scaling(featureDF):
    aggregated = featureDF.agg(F.min(featureDF.distances_bh),F.max(featureDF.distances_bh),F.mean(featureDF.distances_bh),F.stddev(featureDF.distances_bh),
        F.min(featureDF.distances_rh),F.max(featureDF.distances_rh),F.mean(featureDF.distances_rh),F.stddev(featureDF.distances_rh),
        F.min(featureDF.distances_rp),F.max(featureDF.distances_rp),F.mean(featureDF.distances_rp),F.stddev(featureDF.distances_rp),
        F.min(featureDF.distances_corr),F.max(featureDF.distances_corr),F.mean(featureDF.distances_corr),F.stddev(featureDF.distances_corr),
        F.min(featureDF.distances_levenshtein),F.max(featureDF.distances_levenshtein),F.mean(featureDF.distances_levenshtein),F.stddev(featureDF.distances_levenshtein),
        F.min(featureDF.distances_mfcc),F.max(featureDF.distances_mfcc),F.mean(featureDF.distances_mfcc),F.stddev(featureDF.distances_mfcc),
        F.min(featureDF.distances_js),F.max(featureDF.distances_js),F.mean(featureDF.distances_js),F.stddev(featureDF.distances_js),
        F.min(featureDF.distances_skl),F.max(featureDF.distances_skl),F.mean(featureDF.distances_skl),F.stddev(featureDF.distances_skl))
    ##############################
    var_val = aggregated.collect()[0]["stddev_samp(distances_bh)"]
    mean_val = aggregated.collect()[0]["avg(distances_bh)"]
    ##############################
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    featureDF = featureDF.withColumn('scaled_rp', (featureDF.distances_rp-min_val)/(max_val-min_val))
    ##############################    
    max_val = aggregated.collect()[0]["max(distances_rh)"]
    min_val = aggregated.collect()[0]["min(distances_rh)"]
    featureDF = featureDF.withColumn('scaled_rh', (featureDF.distances_rh-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_bh)"]
    min_val = aggregated.collect()[0]["min(distances_bh)"]
    featureDF = featureDF.withColumn('scaled_bh', (featureDF.distances_bh-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_levenshtein)"]
    min_val = aggregated.collect()[0]["min(distances_levenshtein)"]
    featureDF = featureDF.withColumn('scaled_notes', (featureDF.distances_levenshtein-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    featureDF = featureDF.withColumn('scaled_chroma', (1 - (featureDF.distances_corr-min_val)/(max_val-min_val)))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_skl)"]
    min_val = aggregated.collect()[0]["min(distances_skl)"]
    featureDF = featureDF.withColumn('scaled_skl', (featureDF.distances_skl-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_js)"]	
    min_val = aggregated.collect()[0]["min(distances_js)"]
    featureDF = featureDF.withColumn('scaled_js', (featureDF.distances_js-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_mfcc)"]
    min_val = aggregated.collect()[0]["min(distances_mfcc)"]
    featureDF = featureDF.withColumn('scaled_mfcc', (featureDF.distances_mfcc-min_val)/(max_val-min_val)).select("id", "key", "scale", "bpm", "scaled_rp", "scaled_rh", "scaled_bh", "scaled_notes", "scaled_chroma", "scaled_skl", "scaled_js", "scaled_mfcc")
    ##############################
    return featureDF

def get_nearest_neighbors(song, outname, origFeatureDF):
    #get comparison_features
    song = origFeatureDF.filter(origFeatureDF.id == song).collect()
    #create a copy for results without destroying the original
    fullFeatureDF = origFeatureDF.coalesce(repartition_count).persist()
    fullFeatureDF = get_neighbors_rp_euclidean(song, fullFeatureDF).drop('rp')
    fullFeatureDF = get_neighbors_rh_euclidean(song, fullFeatureDF).drop('rh')
    fullFeatureDF = get_neighbors_notes(song, fullFeatureDF).drop('notes')
    fullFeatureDF = get_neighbors_mfcc_euclidean(song, fullFeatureDF).drop('mfccEuc')
    fullFeatureDF = get_neighbors_bh_euclidean(song, fullFeatureDF).drop('bh')
    fullFeatureDF = get_neighbors_mfcc_skl(song, fullFeatureDF)
    fullFeatureDF = get_neighbors_mfcc_js(song, fullFeatureDF).drop('mfccSkl')
    fullFeatureDF = get_neighbors_chroma_corr_valid(song, fullFeatureDF).drop('chroma')
    
    fullFeatureDF = perform_scaling(fullFeatureDF)

    #scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_notes + scaledSim.scaled_rp + scaledSim.scaled_mfcc) / 3)
    fullFeatureDF = fullFeatureDF.withColumn('aggregated', (fullFeatureDF.scaled_notes + fullFeatureDF.scaled_mfcc + fullFeatureDF.scaled_chroma + fullFeatureDF.scaled_bh + fullFeatureDF.scaled_rp + fullFeatureDF.scaled_skl + fullFeatureDF.scaled_js + fullFeatureDF.scaled_rh) / 8)
    fullFeatureDF = fullFeatureDF.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    #scaledSim.show()
    fullFeatureDF.toPandas().to_csv(outname, encoding='utf-8')
    fullFeatureDF.unpersist()

if len (sys.argv) < 2:
    song = "music/Electronic/The XX - Intro.mp3"    #100 testset
    #song = "music/Classical/Katrine_Gislinge-Fr_Elise.mp3"
else: 
    song = sys.argv[1]
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

time_dict = {}
tic1 = int(round(time.time() * 1000))
get_nearest_neighbors(song, "result_group_full_persist.csv", orig_featureDF)
tac1 = int(round(time.time() * 1000))
time_dict['TIME: ']= tac1 - tic1
print time_dict


#once no longer needed, clear orig_featureDF
#orig_featureDF.unpersist()


