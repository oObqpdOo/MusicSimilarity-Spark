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


def get_neighbors_rp_euclidean_rdd_noscale(song):
    #########################################################
    #   Pre- Process RH for Euclidean
    #
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    rp = rp.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    rp = rp.map(lambda x: x.split(';'))
    rp = rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].split(",")))
    kv_rp= rp.map(lambda x: (x[0], list(x[1:])))
    rp_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #  
    comparator = rp_vec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultRH = rp_vec.map(lambda x: (x[0], distance.euclidean(x[1], comparator_value[0])))
    #########################################################
    #   Pre- Process Notes for Levenshtein
    #
    notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
    notes = notes.map(lambda x: x.split(';'))
    notes = notes.map(lambda x: (x[0].replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
    notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[3].replace(',','').replace(' ',''), x[1], x[2]))
    #########################################################
    #   Get Neighbors
    #  
    comparator = notes.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultNotes = notes.map(lambda x: (x[0], naive_levenshtein(x[1], comparator_value[0]), x[1], x[2]))
    #########################################################
    #   Pre- Process MFCC for Euclidean
    #
    mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
    mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcceuc = mfcceuc.map(lambda x: x.split(';'))
    mfcceuc = mfcceuc.map(lambda x: (x[0], x[1].split(',')))
    mfccVec = mfcceuc.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #
    comparator = mfccVec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = Vectors.dense(comparator[0])
    resultMfcc = mfccVec.map(lambda x: (x[0], distance.euclidean(x[1], comparator_value[0])))
    mergedSim = resultMfcc.join(resultNotes)
    mergedSim = mergedSim.join(resultRH)
    mergedSim.toDF().toPandas().to_csv("debug.csv", encoding='utf-8')

def get_neighbors_rp_euclidean_rdd(song):
    #########################################################
    #   Pre- Process RH for Euclidean
    #
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    rp = rp.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    rp = rp.map(lambda x: x.split(';'))
    rp = rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].split(",")))
    kv_rp= rp.map(lambda x: (x[0], list(x[1:])))
    rp_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1])))
    #########################################################
    #   Get Neighbors
    #  
    comparator = rp_vec.lookup(song.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    comparator_value = comparator[0]
    resultRH = rp_vec.map(lambda x: (x[0], distance.euclidean(np.array(x[1]), np.array(comparator_value))))
    max_val = resultRH.max(lambda x:x[1])[1]
    min_val = resultRH.min(lambda x:x[1])[1]  
    resultRH = resultRH.map(lambda x: (x[0], (x[1]-min_val)/(max_val-min_val)))
    return resultRH 

def get_neighbors_notes_rdd(song):
    #########################################################
    #   Pre- Process Notes for Levenshtein
    #
    notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
    notes = notes.map(lambda x: x.split(';'))
    notes = notes.map(lambda x: (x[0].replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
    notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[3].replace(',','').replace(' ',''), x[1], x[2]))
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

def get_neighbors_mfcc_euclidean_rdd(song):
    #########################################################
    #   Pre- Process MFCC for Euclidean
    #
    mfcceuc = sc.textFile("features[0-9]*/out[0-9]*.mfcc")
    mfcceuc = mfcceuc.map(lambda x: x.replace(' ', '').replace('[', '').replace(']', '').replace(']', '').replace(';', ','))
    mfcceuc = mfcceuc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
    mfcceuc = mfcceuc.map(lambda x: x.split(';'))
    mfcceuc = mfcceuc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1].split(',')))
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


def get_neighbors_rp_euclidean_dataframe(song):
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
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances_rp', distance_udf(F.col('features'))).select("id", "distances_rp")
    max_val = result.agg({"distances_rp": "max"}).collect()[0]
    max_val = max_val["max(distances_rp)"]
    min_val = result.agg({"distances_rp": "min"}).collect()[0]
    min_val = min_val["min(distances_rp)"]
    return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")


def get_neighbors_mfcc_euclidean_dataframe(song):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
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
    assembler = VectorAssembler(inputCols=["mean", "var", "cov"],outputCol="features")
    mfccEucDfMerged = assembler.transform(mfccEucDf)
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
    max_val = result.agg({"distances_levenshtein": "max"}).collect()[0]
    max_val = max_val["max(distances_levenshtein)"]
    min_val = result.agg({"distances_levenshtein": "min"}).collect()[0]
    min_val = min_val["min(distances_levenshtein)"]
    return result.withColumn('scaled_levenshtein', (result.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "key", "scale", "scaled_levenshtein")


def get_nearest_neighbors_pregroup(song, outname):
    #########################################################
    #   List to Vector UDF
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    #########################################################
    #   Pre- Process RP for Euclidean
    rp = sc.textFile("features[0-9]*/out[0-9]*.rp")
    rp = rp.map(lambda x: x.split(","))
    kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
    rp_df = spark.createDataFrame(kv_rp, ["id", "rp"])
    rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))
    #########################################################
    #   Pre- Process Notes for Levenshtein
    notes = sc.textFile("features[0-9]*/out[0-9]*.notes")
    notes = notes.map(lambda x: x.split(';'))
    notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
    notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
    notesDf = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])
    #########################################################
    #   Pre- Process MFCC for Euclidean
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
    #########################################################
    #   Gather all features in one dataframe
    #
    featureDF = mfccEucDfMerged.join(rp_df, on=["id"], how='inner')
    featureDF = featureDF.join(notesDf, on=['id'], how='inner').dropDuplicates()
    #print(featureDF.count())
    #featureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')
    #########################################################
    song = featureDF.filter(featureDF.id == song)
    comparator_value_rp = Vectors.dense(song.select("rp").collect()[0][0]) 
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value_rp)), FloatType())
    neighbors_rp_euclidean = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")
    max_val = neighbors_rp_euclidean.agg({"distances_rp": "max"}).collect()[0]
    max_val = max_val["max(distances_rp)"]
    min_val = neighbors_rp_euclidean.agg({"distances_rp": "min"}).collect()[0]
    min_val = min_val["min(distances_rp)"]
    neighbors_rp_euclidean = neighbors_rp_euclidean.withColumn('scaled_rp', (neighbors_rp_euclidean.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")
    #########################################################
    comparator_value_mfcceu = Vectors.dense(song.select("mfccEuc").collect()[0][0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value_mfcceu)), FloatType())
    neighbors_mfcc_eucl = featureDF.withColumn('distances_mfcc', distance_udf(F.col('mfccEuc'))).select("id", "distances_mfcc")
    max_val = neighbors_mfcc_eucl.agg({"distances_mfcc": "max"}).collect()[0]
    max_val = max_val["max(distances_mfcc)"]
    min_val = neighbors_mfcc_eucl.agg({"distances_mfcc": "min"}).collect()[0]
    min_val = min_val["min(distances_mfcc)"]
    neighbors_mfcc_eucl = neighbors_mfcc_eucl.withColumn('scaled_mfcc', (neighbors_mfcc_eucl.distances_mfcc-min_val)/(max_val-min_val)).select("id", "scaled_mfcc")
    #########################################################
    comparator_value_notes = song.select("notes").collect()[0][0]
    df_merged = featureDF.withColumn("compare", lit(comparator_value_notes))
    df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    #df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
    neighbors_notes = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
    max_val = neighbors_notes.agg({"distances_levenshtein": "max"}).collect()[0]
    max_val = max_val["max(distances_levenshtein)"]
    min_val = neighbors_notes.agg({"distances_levenshtein": "min"}).collect()[0]
    min_val = min_val["min(distances_levenshtein)"]
    neighbors_notes = neighbors_notes.withColumn('scaled_levenshtein', (neighbors_notes.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "scaled_levenshtein", "key", "scale")
    #########################################################    
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').dropDuplicates()
    #########################################################
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    #mergedSim.show()
    out_name = outname#"output.csv"
    mergedSim.toPandas().to_csv(out_name, encoding='utf-8')
    return mergedSim

def get_nearest_neighbors_dataframe(song, outname):
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean_dataframe(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_dataframe(song)
    neighbors_notes = get_neighbors_notes_dataframe(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').dropDuplicates()
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)
    mergedSim.toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

def get_nearest_neighbors_rdd(song, outname):
    neighbors_rp_euclidean = get_neighbors_rp_euclidean_rdd(song)
    neighbors_notes = get_neighbors_notes_rdd(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean_rdd(song)
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean)
    mergedSim = mergedSim.join(neighbors_notes)
    #mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    mergedSim = mergedSim.map(lambda x: (x[0], ((x[1][0][1] + x[1][1] + x[1][0][0]) / 3))).sortBy(lambda x: x[1], ascending = True)
    mergedSim.toDF().toPandas().to_csv(outname, encoding='utf-8')
    return mergedSim

#song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
song = "music/Electronic/The XX - Intro.mp3"    #100 testset
song = song.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')

result = get_nearest_neighbors_dataframe(song, "Electro_dataframe.csv")
result.show()

result = get_nearest_neighbors_pregroup(song, "Electro_premerged.csv")
result.show()

result = get_nearest_neighbors_rdd(song, "Electro_rdd.csv")
result.sortBy(lambda x: x[1], ascending = True).take(10)

