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

#mfcc = sc.textFile("features/out0.mfcc")

mfcc = sc.textFile("features/out[0-9]*.mfcckl")
mfcc = mfcc.map(lambda x: x.split(';'))

#key
#print mfcc.first()[0]
#mean
#print mfcc.first()[1]
#covkl
#print mfcc.first()[2]

meanRdd = mfcc.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDf = spark.createDataFrame(meanRdd, ["id", "mean"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
meanVec = meanDf.select(meanDf["id"],list_to_vector_udf(meanDf["mean"]).alias("mean"))
#meanVec.first()

covRdd = mfcc.map(lambda x: (x[0],(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDf = spark.createDataFrame(covRdd, ["id", "cov"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
covVec = covDf.select(covDf["id"],list_to_vector_udf(covDf["cov"]).alias("cov"))
#covVec.first()


mfccDf = meanVec.join(covVec, on=['id'], how='left_outer')

#mfccDf.first()

assembler = VectorAssembler(inputCols=["mean", "cov"],outputCol="features")
mfccDfMerged = assembler.transform(mfccDf)
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
mfccDfMerged.first()


##################################################
# test because of NaN log(negative determinant or zero)

song = "music/THRONES1.mp3"
df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
#df_vec.first()
filterDF = df_vec.filter(df_vec.id == song)
filterDF.first()
comparator_value = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
#print comparator_value
song = "music/THRONES2.mp3"
filterDF = df_vec.filter(df_vec.id == song)
#filterDF.first()
comparator_value2 = comparator_value = Vectors.dense(filterDF.collect()[0][1]) 
print comparator_value2
mean1 = np.empty([13, 1])
mean1 = comparator_value[0:13]
print mean1
cov1 = np.empty([13,13])
cov1 = comparator_value[13:].reshape(13, 13)
print cov1
#get 13 mean and 13x13 cov as vectors
jensen_shannon(comparator_value, comparator_value2)

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

song = "music/PUNISH2.mp3"
get_neighbors_mfcc_js(song)
get_neighbors_mfcc_skl(song)

song = "music/THRONES2.mp3"
get_neighbors_mfcc_js(song)
get_neighbors_mfcc_skl(song)
