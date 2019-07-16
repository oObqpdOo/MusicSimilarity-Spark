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
from scipy.signal import butter, lfilter, freqz, correlate2d

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("cluster").appName("MusicSimilarity").getOrCreate()

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
#   Pre- Process RP for Euclidean
#
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
assembler = VectorAssembler(inputCols=["mean", "cov"],outputCol="features")
mfccDfMerged = assembler.transform(mfccDf)
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
#mfccDfMerged.first()

#########################################################
#   Pre- Process Chroma for cross-correlation
#

chroma = sc.textFile("features[0-9]*/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))
chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaDf = spark.createDataFrame(chromaRdd, ["id", "chroma"])
chromaVec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))

#########################################################
#   Pre- Process MFCC for Euclidean
#

mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0], list(x[1:])))

meanRddEuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1][0].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDfEuc = spark.createDataFrame(meanRddEuc, ["id", "mean"])
meanVecEuc = meanDfEuc.select(meanDfEuc["id"],list_to_vector_udf(meanDfEuc["mean"]).alias("mean"))

varRddEuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1][1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
varDfEuc = spark.createDataFrame(varRddEuc, ["id", "var"])
varVecEuc = varDfEuc.select(varDfEuc["id"],list_to_vector_udf(varDfEuc["var"]).alias("var"))

covRddEuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""),(x[1][2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDfEuc = spark.createDataFrame(covRddEuc, ["id", "cov"])
covVecEuc = covDfEuc.select(covDfEuc["id"],list_to_vector_udf(covDfEuc["cov"]).alias("cov"))

mfccEucDf = meanVecEuc.join(varVecEuc, on=['id'], how='inner')
mfccEucDf = mfccEucDf.join(covVecEuc, on=['id'], how='inner').dropDuplicates()

assembler = VectorAssembler(inputCols=["mean", "var", "cov"],outputCol="features")
mfccEucDfMerged = assembler.transform(mfccEucDf)


def get_neighbors_chroma_corr_valid(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    max_val = result.agg({"distances_corr": "max"}).collect()[0]
    max_val = max_val["max(distances_corr)"]
    min_val = result.agg({"distances_corr": "min"}).collect()[0]
    min_val = min_val["min(distances_corr)"]
    return result.withColumn('scaled_corr', 1 - (result.distances_corr-min_val)/(max_val-min_val)).select("id", "scaled_corr")



def get_neighbors_mfcc_euclidean(song):
    df_vec = mfccEucDfMerged.select(mfccEucDfMerged["id"],list_to_vector_udf(mfccEucDfMerged["features"]).alias("features"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_mfcc', distance_udf(F.col('features'))).select("id", "distances_mfcc")
    max_val = result.agg({"distances_mfcc": "max"}).collect()[0]
    max_val = max_val["max(distances_mfcc)"]
    min_val = result.agg({"distances_mfcc": "min"}).collect()[0]
    min_val = min_val["min(distances_mfcc)"]
    return result.withColumn('scaled_mfcc', (result.distances_mfcc-min_val)/(max_val-min_val)).select("id", "scaled_mfcc")



def get_neighbors_mfcc_skl(song):
    df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances_skl', distance_udf(F.col('features'))).select("id", "distances_skl")
    #thresholding 
    #result = result.filter(result.distances_skl <= 1000)  
    max_val = result.agg({"distances_skl": "max"}).collect()[0]
    max_val = max_val["max(distances_skl)"]
    min_val = result.agg({"distances_skl": "min"}).collect()[0]
    min_val = min_val["min(distances_skl)"]
    return result.withColumn('scaled_skl', (result.distances_skl-min_val)/(max_val-min_val)).select("id", "scaled_skl")

def get_neighbors_rp_euclidean(song):
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_rp', distance_udf(F.col('features'))).select("id", "distances_rp")
    max_val = result.agg({"distances_rp": "max"}).collect()[0]
    max_val = max_val["max(distances_rp)"]
    min_val = result.agg({"distances_rp": "min"}).collect()[0]
    min_val = min_val["min(distances_rp)"]
    return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")


def get_neighbors_notes(song):
    df = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])
    filterDF = df.filter(df.id == song)
    comparator_value = filterDF.collect()[0][3] 
    df_merged = df.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    #df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
    result = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
    max_val = result.agg({"distances_levenshtein": "max"}).collect()[0]
    max_val = max_val["max(distances_levenshtein)"]
    min_val = result.agg({"distances_levenshtein": "min"}).collect()[0]
    min_val = min_val["min(distances_levenshtein)"]
    return result.withColumn('scaled_levenshtein', (result.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "key", "scale", "scaled_levenshtein")


def get_neighbors_bh_euclidean(song):
    df = spark.createDataFrame(kv_bh, ["id", "bpm", "features"])
    filterDF = df.filter(df.id == song)
    comparator_value = filterDF.collect()[0][2]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df.withColumn('distances_bh', distance_udf(F.col('features'))).select("id", "bpm", "distances_bh")
    max_val = result.agg({"distances_bh": "max"}).collect()[0]
    max_val = max_val["max(distances_bh)"]
    min_val = result.agg({"distances_bh": "min"}).collect()[0]
    min_val = min_val["min(distances_bh)"]
    return result.withColumn('scaled_bh', (result.distances_bh-min_val)/(max_val-min_val)).select("id", "bpm", "scaled_bh")



def get_nearest_neighbors(song, outname):
    neighbors_mfcc_skl = get_neighbors_mfcc_skl(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_bh_euclidean = get_neighbors_bh_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)

    mergedSim = neighbors_rp_euclidean.join(neighbors_notes, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_eucl, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_mfcc_skl, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_corr + mergedSim.scaled_mfcc + mergedSim.scaled_skl + mergedSim.scaled_bh) / 6)
    #mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')


#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3".replace(";","").replace(".","").replace(",","").replace(" ","")    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Oldschool/Stranger Things (Soundtrack)/26 - Rock You Like a Hurricane [Explicit].mp3".encode('utf-8','replace').replace(";","").replace(".","").replace(",","").replace(" ","")    #100 testset


songs = sc.textFile("features[0-9]*/out.files", use_unicode=True)
list1 = songs.map(lambda x: x.split(':'))
#DO NOT USE str(x[0]) USE x[0].encode('utf-8') instead!!
list1 = list1.map(lambda x: x[0])
list1 = list1.map(lambda x: x.replace(";","").replace(".","").replace(",","").replace(" ",""))
list1l = list1.collect()

for i in list1l[51:100]: 
    #outname = str(i).encode('utf-8','replace').replace('.mp3', '').replace('music/', '').rpartition('/')[0] + ".csv"
    outname = i.replace('.mp3', '').replace('music/', '').replace('/', '_').replace('mp3', '') + ".csv"
    #has to be encoded back to ascii string to print    
    outname = outname.encode('ascii','ignore')    
    print outname 
    get_nearest_neighbors(i, outname)



#song = "music/Reggae/Damian Marley - Confrontation.mp3"
#get_nearest_neighbors_fast(song, "Reggae_fast.csv")
#song = "music/Reggae/Damian Marley - Confrontation.mp3"
#get_nearest_neighbors_precise(song, "Reggae_precise.csv")
#song = "music/Reggae/Damian Marley - Confrontation.mp3"
#get_nearest_neighbors_full(song, "Reggae_full.csv")




#song = "music/Soundtrack/Flesh And Bone - Dakini_ Movement IV.mp3"
#get_nearest_neighbors_fast(song, "Soundtrack_fast.csv")
#song = "music/Hip Hop/Kid Cudi - Mr Rager.mp3"
#get_nearest_neighbors_fast(song, "HipHop_fast.csv")
#song = "music/Metal/Gojira - Global warming.mp3"
#get_nearest_neighbors_fast(song, "Metal_fast.csv")
#song = "music/Electronic/Dynatron - Pulse Power.mp3"
#get_nearest_neighbors_fast(song, "Electronic 2_fast.csv")

#song = "music/Soundtrack/Flesh And Bone - Dakini_ Movement IV.mp3"
#get_nearest_neighbors_precise(song, "Soundtrack_precise.csv")
#song = "music/Hip Hop/Kid Cudi - Mr Rager.mp3"
#get_nearest_neighbors_precise(song, "HipHop_precise.csv")
#song = "music/Metal/Gojira - Global warming.mp3"
#get_nearest_neighbors_precise(song, "Metal_precise.csv")
#song = "music/Electronic/Dynatron - Pulse Power.mp3"
#get_nearest_neighbors_precise(song, "Electronic 2_precise.csv")

#song = "music/Soundtrack/Flesh And Bone - Dakini_ Movement IV.mp3"
#get_nearest_neighbors_full(song, "Soundtrack_full.csv")
#song = "music/Hip Hop/Kid Cudi - Mr Rager.mp3"
#get_nearest_neighbors_full(song, "HipHop_full.csv")
#song = "music/Metal/Gojira - Global warming.mp3"
#get_nearest_neighbors_full(song, "Metal_full.csv")
#song = "music/Electronic/Dynatron - Pulse Power.mp3"
#get_nearest_neighbors_full(song, "Electronic 2_full.csv")




