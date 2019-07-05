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
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
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
from pyspark.sql import SQLConf


confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)

song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"    #private
#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"           #1517 artists
#song = "music/HURRICANE1.mp3"              #small testset

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

#########################################################
#   Pre- Process RH and RP for Euclidean
#

rp = sc.textFile("features/out[0-9]*.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0], list(x[1:])))

#########################################################
#   Pre- Process Notes for Levenshtein
#

notes = sc.textFile("features/out[0-9]*.notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))

#########################################################
#   Pre- Process Chroma for cross-correlation
#

from pyspark.sql.column import _to_java_column, _to_seq, Column
from pyspark import SparkContext

def as_vector(col):
    sc = SparkContext.getOrCreate()
    f = sc._jvm.com.example.spark.udfs.udfs.as_vector()
    return Column(f.apply(_to_seq(sc, [col], _to_java_column)))

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
list_to_vector_udf = udf(lambda l: Vectors.dense(l))


l = [('Alice', 1)]
sqlContext.createDataFrame(l).collect()
sqlContext.createDataFrame(l, ['name', 'age']).collect()

chroma = sc.textFile("features/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))
chromaRdd = chroma.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaRdd = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
chromaRdd.first()
chromaDf = sqlContext.createDataFrame(chromaRdd, ["id", "chroma"])
chromaVec = chromaDf

#########################################################
#   Pre- Process MFCC for SKL and JS
#

mfcc = sc.textFile("features/out[0-9]*.mfcckl")
mfcc = mfcc.map(lambda x: x.split(';'))

meanRdd = mfcc.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDf = sqlContext.createDataFrame(meanRdd, ["id", "mean"])
meanVec = meanDf.select(meanDf["id"],list_to_vector_udf(meanDf["mean"]).alias("mean"))
#meanVec.first()

covRdd = mfcc.map(lambda x: (x[0],(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDf = sqlContext.createDataFrame(covRdd, ["id", "cov"])
covVec = covDf.select(covDf["id"],list_to_vector_udf(covDf["cov"]).alias("cov"))
#covVec.first()

mfccDf = meanVec.join(covVec, on=['id'], how='left_outer')
assembler = VectorAssembler(inputCols=["mean", "cov"],outputCol="features")
mfccDfMerged = assembler.transform(mfccDf)
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
#mfccDfMerged.first()



def get_neighbors_mfcc_skl(song):
    df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
    #df_vec.first()
    filterDF = df_vec.filter(df_vec.id == song)
    #filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    #print result
    return result



def get_neighbors_rp_euclidean(song):
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = sqlContext.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    #print result
    return result



def get_neighbors_notes(song):
    df = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])
    filterDF = df.filter(df.id == song)
    #filterDF.first()
    comparator_value = filterDF.collect()[0][3] 
    df_merged = df.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("word1_word2_levenshtein", levenshtein(col("notes"), col("compare")))
    #df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
    return df_levenshtein.sort(col("word1_word2_levenshtein").asc())



def get_neighbors_chroma_corr_valid(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    filterDF = df_vec.filter(df_vec.id == song)
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('chroma')))
    result = result.select("id", "distances").orderBy('distances', ascending=False)
    result = result.rdd.flatMap(list).collect()
    #print result
    return result



def get_nearest_neighbors(song):
    neighbors_mfcc_skl = get_neighbors_mfcc_skl(song)
    neighbors_rp_euclidean = get_neighbors_rp_euclidean(song)
    neighbors_notes = get_neighbors_notes(song)
    neighbors_chroma = get_neighbors_chroma_corr(song)
    print neighbors_mfcc_skl[:10]
    print neighbors_rp_euclidean[:10]
    neighbors_notes.show()
    print neighbors_chroma[:10]


#song = "music/Rock & Pop/Sabaton-Primo_Victoria.mp3"
#song = "music/HURRICANE1.mp3"
song = "music/Jazz & Klassik/Keith Jarret - Creation/02-Keith Jarrett-Part II Tokyo.mp3"  

get_nearest_neighbors(song)



