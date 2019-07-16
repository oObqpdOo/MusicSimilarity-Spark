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
    transposed_chroma = transposed_chroma /(min(length1, length2))
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



#########################################################
#   List to Vector UDF
#

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

#########################################################
#   Pre- Process RH and RP for Euclidean
#
rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rp_df = spark.createDataFrame(kv_rp, ["id", "rp"])
rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))
rh = sc.textFile("features[0-9]*/out[0-9]*.rh")
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rh_df = spark.createDataFrame(kv_rh, ["id", "rh"])
rh_df = rh_df.select(rh_df["id"],list_to_vector_udf(rh_df["rh"]).alias("rh"))

#########################################################
#   Pre- Process BH for Euclidean
#
bh = sc.textFile("features[0-9]*/out[0-9]*.bh")
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
bh_df = spark.createDataFrame(kv_bh, ["id", "bpm", "bh"])

#########################################################
#   Pre- Process Notes for Levenshtein
#
notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
notesDf = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])

#########################################################
#   Pre- Process Chroma for cross-correlation
#
chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))
chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaDf = spark.createDataFrame(chromaRdd, ["id", "chroma"])
chromaDf = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))

#########################################################
#   Pre- Process MFCC for SKL and JS
#
mfcc = sc.textFile("features[0-9]*/out[0-9]*.mfcckl")
mfcc = mfcc.map(lambda x: x.split(';'))
meanRdd = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDf = spark.createDataFrame(meanRdd, ["id", "mean"])
meanVec = meanDf.select(meanDf["id"],list_to_vector_udf(meanDf["mean"]).alias("mean"))
#meanVec.first()
covRdd = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDf = spark.createDataFrame(covRdd, ["id", "cov"])
covVec = covDf.select(covDf["id"],list_to_vector_udf(covDf["cov"]).alias("cov"))
#covVec.first()
mfccDf = meanVec.join(covVec, on=['id'], how='inner').dropDuplicates()
assembler = VectorAssembler(inputCols=["mean", "cov"],outputCol="mfccSkl")
mfccDfMerged = assembler.transform(mfccDf).select("id", "mfccSkl").dropDuplicates()
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
#mfccDfMerged.first()

#########################################################
#   Pre- Process MFCC for Euclidean
#
mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
meanRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][0].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDfEuc = spark.createDataFrame(meanRddEuc, ["id", "mean"])
meanVecEuc = meanDfEuc.select(meanDfEuc["id"],list_to_vector_udf(meanDfEuc["mean"]).alias("mean"))
varRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
varDfEuc = spark.createDataFrame(varRddEuc, ["id", "var"])
varVecEuc = varDfEuc.select(varDfEuc["id"],list_to_vector_udf(varDfEuc["var"]).alias("var"))
covRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDfEuc = spark.createDataFrame(covRddEuc, ["id", "cov"])
covVecEuc = covDfEuc.select(covDfEuc["id"],list_to_vector_udf(covDfEuc["cov"]).alias("cov"))
mfccEucDf = meanVecEuc.join(varVecEuc, on=['id'], how='inner')
mfccEucDf = mfccEucDf.join(covVecEuc, on=['id'], how='inner').dropDuplicates()
assembler = VectorAssembler(inputCols=["mean", "var", "cov"],outputCol="mfccEuc")
mfccEucDfMerged = assembler.transform(mfccEucDf).select("id", "mfccEuc").dropDuplicates()

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
featureDF = mfccEucDfMerged.join(mfccDfMerged, on=["id"], how='inner')
featureDF = featureDF.join(chromaDf, on=['id'], how='inner')
featureDF = featureDF.join(notesDf, on=['id'], how='inner')
featureDF = featureDF.join(rp_df, on=['id'], how='inner')
featureDF = featureDF.join(rh_df, on=['id'], how='inner')
featureDF = featureDF.join(bh_df, on=['id'], how='inner').dropDuplicates()
#print(featureDF.count())
fullFeatureDF = featureDF
#fullFeatureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')

#featureDF.saveAsTable("featureDF")
#song = "music/Reggae/Damian Marley - Confrontation.mp3"
#filterDF = featureDF.filter(featureDF.id == song)
#comparator_value_ = Vectors.dense(filterDF.select("mfccEuc").collect())


def get_neighbors_mfcc_skl(song, featureDF):
    comparator_value = Vectors.dense(song.select("mfccSkl").collect()[0][0]) 
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_skl', distance_udf(F.col('mfccSkl'))).select("id", "distances_skl")
    #thresholding 
    result = result.filter(result.distances_skl <= 100)  
    return result

def get_neighbors_mfcc_js(song, featureDF):
    comparator_value = Vectors.dense(song.select("mfccSkl").collect()[0][0])   
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl'))).select("id", "distances_js")
    result = result.filter(result.distances_js != np.inf)    
    return result

def get_neighbors_rp_euclidean(song, featureDF):
    comparator_value = Vectors.dense(song.select("rp").collect()[0][0]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")
    return result

def get_neighbors_rh_euclidean(song, featureDF):
    comparator_value = Vectors.dense(song.select("rh").collect()[0][0]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_rh', distance_udf(F.col('rh'))).select("id", "distances_rh")
    return result

def get_neighbors_bh_euclidean(song, featureDF):
    comparator_value = Vectors.dense(song.select("bh").collect()[0][0]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_bh', distance_udf(F.col('bh'))).select("id", "bpm", "distances_bh")
    return result

def get_neighbors_mfcc_euclidean(song, featureDF):
    comparator_value = Vectors.dense(song.select("mfccEuc").collect()[0][0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_mfcc', distance_udf(F.col('mfccEuc'))).select("id", "distances_mfcc")
    return result

def get_neighbors_notes(song, featureDF):
    comparator_value = song.select("notes").collect()[0][0]
    df_merged = featureDF.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    #df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
    result = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
    return result

def get_neighbors_chroma_corr_valid(song, featureDF):
    comparator_value = Vectors.dense(song.select("chroma").collect()[0][0])
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_full(x, comparator_value)), DoubleType())
    result = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    return result

def perform_scaling(unscaled_df):
    aggregated = unscaled_df.agg(F.min(unscaled_df.distances_bh),F.max(unscaled_df.distances_bh),F.mean(unscaled_df.distances_bh),F.stddev(unscaled_df.distances_bh),
        F.min(unscaled_df.distances_rh),F.max(unscaled_df.distances_rh),F.mean(unscaled_df.distances_rh),F.stddev(unscaled_df.distances_rh),
        F.min(unscaled_df.distances_rp),F.max(unscaled_df.distances_rp),F.mean(unscaled_df.distances_rp),F.stddev(unscaled_df.distances_rp),
        F.min(unscaled_df.distances_bh),F.max(unscaled_df.distances_bh),F.mean(unscaled_df.distances_bh),F.stddev(unscaled_df.distances_bh),
        F.min(unscaled_df.distances_corr),F.max(unscaled_df.distances_corr),F.mean(unscaled_df.distances_corr),F.stddev(unscaled_df.distances_corr),
        F.min(unscaled_df.distances_levenshtein),F.max(unscaled_df.distances_levenshtein),F.mean(unscaled_df.distances_levenshtein),F.stddev(unscaled_df.distances_levenshtein),
        F.min(unscaled_df.distances_mfcc),F.max(unscaled_df.distances_mfcc),F.mean(unscaled_df.distances_mfcc),F.stddev(unscaled_df.distances_mfcc),
        F.min(unscaled_df.distances_js),F.max(unscaled_df.distances_js),F.mean(unscaled_df.distances_js),F.stddev(unscaled_df.distances_js),
        F.min(unscaled_df.distances_skl),F.max(unscaled_df.distances_skl),F.mean(unscaled_df.distances_skl),F.stddev(unscaled_df.distances_skl))
    ##############################
    var_val = aggregated.collect()[0]["stddev_samp(distances_bh)"]
    mean_val = aggregated.collect()[0]["avg(distances_bh)"]
    ##############################
    max_val = aggregated.collect()[0]["max(distances_rp)"]
    min_val = aggregated.collect()[0]["min(distances_rp)"]
    result = unscaled_df.withColumn('scaled_rp', (unscaled_df.distances_rp-min_val)/(max_val-min_val)).select("id", "key", "scale", "bpm", "scaled_rp")
    ##############################    
    max_val = aggregated.collect()[0]["max(distances_rh)"]
    min_val = aggregated.collect()[0]["min(distances_rh)"]
    result = result.withColumn('scaled_rh', (unscaled_df.distances_rh-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_bh)"]
    min_val = aggregated.collect()[0]["min(distances_bh)"]
    result = result.withColumn('scaled_bh', (unscaled_df.distances_bh-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_levenshtein)"]
    min_val = aggregated.collect()[0]["min(distances_levenshtein)"]
    result = result.withColumn('scaled_notes', (unscaled_df.distances_levenshtein-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_corr)"]
    min_val = aggregated.collect()[0]["min(distances_corr)"]
    result = result.withColumn('scaled_chroma', (unscaled_df.distances_corr-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_skl)"]
    min_val = aggregated.collect()[0]["min(distances_skl)"]
    result = result.withColumn('scaled_skl', (unscaled_df.distances_skl-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_js)"]
    min_val = aggregated.collect()[0]["min(distances_js)"]
    result = result.withColumn('scaled_js', (unscaled_df.distances_js-min_val)/(max_val-min_val))
    ##############################
    max_val = aggregated.collect()[0]["max(distances_mfcc)"]
    min_val = aggregated.collect()[0]["min(distances_mfcc)"]
    result = result.withColumn('scaled_mfcc', (unscaled_df.distances_mfcc-min_val)/(max_val-min_val))
    ##############################
    return result

def get_nearest_neighbors(song, outname):
    song = fullFeatureDF.filter(featureDF.id == song)

    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song, fullFeatureDF)
    neighbors_rh_euclidean = get_neighbors_rh_euclidean(song, fullFeatureDF)    
    neighbors_notes = get_neighbors_notes(song, fullFeatureDF)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song, fullFeatureDF)
    neighbors_bh_euclidean = get_neighbors_bh_euclidean(song, fullFeatureDF)
    neighbors_mfcc_skl = get_neighbors_mfcc_skl(song, fullFeatureDF)
    neighbors_mfcc_js = get_neighbors_mfcc_js(song, fullFeatureDF)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song, fullFeatureDF)
    
    #print neighbors_mfcc_skl.first()
    #print neighbors_rp_euclidean.first()
    #neighbors_notes.show()
    #JOIN could also left_inner and handle 'nones'
   
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_rh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_skl, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_js, on=['id'], how='inner').dropDuplicates()

    mergedSim = perform_scaling(mergedSim)
    
    #mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_mfcc + mergedSim.scaled_corr + mergedSim.scaled_bh + mergedSim.scaled_rp + mergedSim.scaled_skl + mergedSim.scaled_js + mergedSim.scaled_rh) / 8)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    mergedSim.show()
    out_name = outname#"output.csv"
    mergedSim.toPandas().to_csv(out_name, encoding='utf-8')


song = "music/Electronic/The XX - Intro.mp3"    #100 testset
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
get_nearest_neighbors(song, "Electro_pregroup_full.csv")

