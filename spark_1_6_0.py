#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.ml.linalg
import pyspark.mllib.param
from scipy.spatial import distance
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d

#from pyspark import SparkContext, SparkConf
#from pyspark.sql import SQLContext, Row
#confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
#confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
#sc = SparkContext(conf=confCluster)
#sqlContext = SQLContext(sc)

song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Klassik Solo/JS Bach_ Toccata And Fugue In D Minor BWV 565 - Toccata.mp3"
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
#song = "music/HURRICANE1.mp3"              #small testset

#########################################################
#   Pre- Process RH and RP for Euclidean
#

rp = sc.textFile("features/out[0-9]*.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0], list(x[1:])))


##############

comparator = kv_rp.lookup(song)
comparator_value = comparator[0]
comparator_value = Vectors.dense(comparator[0])

rp_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1])))



max_val = result.agg({"distances_rp": "max"}).collect()[0]
max_val = max_val["max(distances_rp)"]
min_val = result.agg({"distances_rp": "min"}).collect()[0]
min_val = min_val["min(distances_rp)"]
return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")



#########################################################
#   Pre- Process BH for Euclidean
#

bh = sc.textFile("features/out[0-9]*.bh")
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0], x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))

#########################################################
#   Pre- Process Notes for Levenshtein
#

notes = sc.textFile("features/out[0-9]*.notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))

df = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])
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
return result.withColumn('scaled_levenshtein', (result.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "scaled_levenshtein")


#########################################################
#   Pre- Process MFCC for Euclidean
#

mfcceuc = sc.textFile("features/out[0-9]*.mfcc")
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0], list(x[1:])))

meanRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][0].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanVecEuc = meanRddEuc.map(lambda x: (x[0], Vectors.dense(x[1])))

varRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
varVecEuc = varRddEuc.map(lambda x: (x[0], Vectors.dense(x[1])))

covRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covVecEuc = covRddEuc.map(lambda x: (x[0], Vectors.dense(x[1])))

mfccEuc = meanVecEuc.join(varVecEuc, on=['id'], how='left_outer')
mfccEuc = mfccEuc.join(covVecEuc, on=['id'], how='left_outer')

def get_neighbors_rp_euclidean(song):
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df_vec = kv_rp.map(lambda x: (x[0], Vectors.dense(x[1]))))
    comparator_value = Vectors.dense(comparator[0])

    result = df_vec.withColumn('distances_rp', distance_udf(F.col('features'))).select("id", "distances_rp")
        
    max_val = result.agg({"distances_rp": "max"}).collect()[0]
    max_val = max_val["max(distances_rp)"]
    min_val = result.agg({"distances_rp": "min"}).collect()[0]
    min_val = min_val["min(distances_rp)"]
    return result.withColumn('scaled_rp', (result.distances_rp-min_val)/(max_val-min_val)).select("id", "scaled_rp")



def get_neighbors_notes(song):
    df = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])
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
    return result.withColumn('scaled_levenshtein', (result.distances_levenshtein-min_val)/(max_val-min_val)).select("id", "scaled_levenshtein")



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
    return result.withColumn('scaled_bh', (result.distances_bh-min_val)/(max_val-min_val)).select("id", "scaled_bh")



def get_nearest_neighbors(song, outname):
    #neighbors_mfcc_skl = get_neighbors_mfcc_skl(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    #neighbors_chroma = get_neighbors_chroma_corr_valid(song)
    neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song)
    #neighbors_bh_euclidean = get_neighbors_bh_euclidean(song)
    #print neighbors_mfcc_skl.first()
    #print neighbors_rp_euclidean.first()
    #neighbors_notes.show()
    #JOIN could also left_inner and handle 'nones'
    mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner')
    #mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')
    #mergedSim = mergedSim.join(neighbors_mfcc_eucl, on=['id'], how='inner')
    #mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')
    mergedSim = mergedSim.withColumn('aggregated', (mergedSim.scaled_levenshtein + mergedSim.scaled_rp + mergedSim.scaled_mfcc) / 3)
    mergedSim = mergedSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
    mergedSim.show()
    out_name = outname#"output.csv"
    mergedSim.toPandas().to_csv(out_name, encoding='utf-8')

song = "music/Klassik Solo/JS Bach_ Toccata And Fugue In D Minor BWV 565 - Toccata.mp3"
get_nearest_neighbors(song, "Toccata.csv")

song = "music/Black Metal/Emperor - I Am The Black Wizards.mp3"
get_nearest_neighbors(song, "BlackMetal.csv")

song = "music/Melodic Death Metal/In Flames - The New Word.mp3"
get_nearest_neighbors(song, "Melodeath.csv")

song = "music/Electronic/The XX - Intro.mp3"
get_nearest_neighbors(song, "Electro.csv")

song = "music/Hip Hop German/Sido - Spring rauf.mp3"
get_nearest_neighbors(song, "HipHop.csv")



