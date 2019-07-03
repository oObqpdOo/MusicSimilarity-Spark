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
from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
import org.apache.spark.sql.functions.typedLit
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
import scipy as sc
from scipy.signal import butter, lfilter, freqz, correlate2d

def chroma_cross_correlate(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([length1,12])
    chroma1 = chroma1_par.reshape(length1, 12)
    length2 = chroma2_par.size/12
    chroma2 = np.empty([length2,12])
    chroma2 = chroma2_par.reshape(length2, 12)
    corr = sc.signal.correlate2d(chroma1, chroma2, mode='full') 
    transposed_chroma = np.transpose(corr)
    mean_line = transposed_chroma[12]
    #print np.max(mean_line)
    return np.max(mean_line)

def chroma_cross_correlate_full(chroma1_par, chroma2_par):
    length1 = chroma1_par.size/12
    chroma1 = np.empty([length1,12])
    chroma1 = chroma1_par.reshape(length1, 12)
    length2 = chroma2_par.size/12
    chroma2 = np.empty([length2,12])
    chroma2 = chroma2_par.reshape(length2, 12)
    corr = sc.signal.correlate2d(chroma1, chroma2, mode='full')
    #transposed_chroma = np.transpose(transposed_chroma)
    #mean_line = transposed_chroma[12]
    #print np.max(corr)
    return np.max(corr)

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

song = "music/PUNISH2.mp3"

#########################################################
#   Pre- Process RH and RP for Euclidean
#

rh = sc.textFile("features/out.rh")
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0], list(x[1:])))

rp = sc.textFile("features/out.rp")
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

chroma = sc.textFile("features/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))
chromaRdd = chroma.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaDf = spark.createDataFrame(chromaRdd, ["id", "chroma"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
chromaVec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))


#########################################################
#   Pre- Process MFCC for SKL and JS
#

mfcc = sc.textFile("features/out[0-9]*.mfcckl")
mfcc = mfcc.map(lambda x: x.split(';'))

meanRdd = mfcc.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDf = spark.createDataFrame(meanRdd, ["id", "mean"])
meanVec = meanDf.select(meanDf["id"],list_to_vector_udf(meanDf["mean"]).alias("mean"))
#meanVec.first()

covRdd = mfcc.map(lambda x: (x[0],(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDf = spark.createDataFrame(covRdd, ["id", "cov"])
covVec = covDf.select(covDf["id"],list_to_vector_udf(covDf["cov"]).alias("cov"))
#covVec.first()

mfccDf = meanVec.join(covVec, on=['id'], how='left_outer')
assembler = VectorAssembler(inputCols=["mean", "cov"],outputCol="features")
mfccDfMerged = assembler.transform(mfccDf)
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
mfccDfMerged.first()

#########################################################
#   Pre- Process MFCC for Euclidean
#

mfcceuc = sc.textFile("features/out[0-9]*.mfcc")
mfcceuc = mfcceuc.map(lambda x: x.split(';'))
mfcceuc = mfcceuc.map(lambda x: (x[0], list(x[1:])))

meanRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][0].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDfEuc = spark.createDataFrame(meanRddEuc, ["id", "mean"])
meanVecEuc = meanDfEuc.select(meanDfEuc["id"],list_to_vector_udf(meanDfEuc["mean"]).alias("mean"))

varRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
varDfEuc = spark.createDataFrame(varRddEuc, ["id", "var"])
varVecEuc = varDfEuc.select(varDfEuc["id"],list_to_vector_udf(varDfEuc["var"]).alias("var"))

covRddEuc = mfcceuc.map(lambda x: (x[0],(x[1][2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDfEuc = spark.createDataFrame(covRddEuc, ["id", "cov"])
covVecEuc = covDfEuc.select(covDfEuc["id"],list_to_vector_udf(covDfEuc["cov"]).alias("cov"))

mfccEucDf = meanVecEuc.join(varVecEuc, on=['id'], how='left_outer')
mfccEucDf = mfccEucDf.join(covVecEuc, on=['id'], how='left_outer')

assembler = VectorAssembler(inputCols=["mean", "var", "cov"],outputCol="features")
mfccEucDfMerged = assembler.transform(mfccEucDf)

def get_neighbors_mfcc_js(song):
    df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
    #df_vec.first()
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(jensen_shannon(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result
    
def get_neighbors_mfcc_skl(song):
    df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
    #df_vec.first()
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result

def get_neighbors_rh_brp(song):
    comparator = kv_rh.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rh, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
    model = brp.fit(df_vec)
    comparator_value = Vectors.dense(comparator[0])
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result

def get_neighbors_rp_brp(song):
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
    model = brp.fit(df_vec)
    comparator_value = Vectors.dense(comparator[0])
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result

def get_neighbors_rh_euclidean(song):
    comparator = kv_rh.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rh, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result

def get_neighbors_rp_euclidean(song):
    comparator = kv_rp.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rp, ["id", "features"])
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result

def get_neighbors_notes(song):
    df = spark.createDataFrame(notes, ["id", "key", "scale", "notes"])
    filterDF = df.filter(df.id == song)
    filterDF.first()
    comparator_value = filterDF.collect()[0][3] 
    print comparator_value
    df_merged = df.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("word1_word2_levenshtein", levenshtein(col("notes"), col("compare")))
    df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()
        
def get_neighbors_chroma_corr_full(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    #df_vec.first()
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_full(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('chroma')))
    result = result.select("id", "distances").orderBy('distances', ascending=False)
    result = result.rdd.flatMap(list).collect()
    print result
    
def get_neighbors_chroma_corr(song):
    df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
    #df_vec.first()
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate(x, comparator_value)), DoubleType())
    result = df_vec.withColumn('distances', distance_udf(F.col('chroma')))
    result = result.select("id", "distances").orderBy('distances', ascending=False)
    result = result.rdd.flatMap(list).collect()
    print result

def get_neighbors_mfcc_brp(song):
    df_vec = mfccEucDfMerged.select(mfccEucDfMerged["id"],list_to_vector_udf(mfccEucDfMerged["features"]).alias("features"))
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
    model = brp.fit(df_vec)
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result   
    #result = model.approxSimilarityJoin(df_vec, df_vec, np.inf, distCol="EuclideanDistance")
    #print result.collect()

def get_neighbors_mfcc_euclidean(song):
    df_vec = mfccEucDfMerged.select(mfccEucDfMerged["id"],list_to_vector_udf(mfccEucDfMerged["features"]).alias("features"))
    #df_ved.first()
    filterDF = df_vec.filter(df_vec.id == song)
    filterDF.first()
    comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
    #print comparator_value
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result

song = "music/PUNISH2.mp3"
get_neighbors_mfcc_js(song)
get_neighbors_mfcc_skl(song)
song = "music/THRONES2.mp3"
get_neighbors_mfcc_js(song)
get_neighbors_mfcc_skl(song)

song = "music/PUNISH2.mp3"
get_neighbors_rh_brp(song)
get_neighbors_rh_euclidean(song)
get_neighbors_rp_brp(song)
get_neighbors_rp_euclidean(song)

song = "music/TURCA1.wav"
get_neighbors_notes(song)
song = "music/RACH1.mp3"
get_neighbors_notes(song)
song = "music/TNT1.mp3"
get_neighbors_notes(song)

song = "music/HURRICANE1.mp3"
get_neighbors_chroma_corr_full(song)
get_neighbors_chroma_corr(song)
song = "music/TNT1.mp3"
get_neighbors_chroma_corr_full(song)
get_neighbors_chroma_corr(song)
song = "music/PUNISH1.mp3"
get_neighbors_chroma_corr_full(song)
get_neighbors_chroma_corr(song)
song = "music/AFRICA1.mp3"
get_neighbors_chroma_corr_full(song)
get_neighbors_chroma_corr(song)

song = "music/AFRICA1.mp3"
get_neighbors_mfcc_brp(song)
get_neighbors_mfcc_euclidean(song)


