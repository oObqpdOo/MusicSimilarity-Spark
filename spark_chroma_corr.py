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
import scipy as sc
from scipy.signal import butter, lfilter, freqz, correlate2d

song = "music/PUNISH2.mp3"

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





chroma = sc.textFile("features/out[0-9]*.chroma")
chroma = chroma.map(lambda x: x.split(';'))

chromaRdd = chroma.map(lambda x: (x[0],(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaDf = spark.createDataFrame(chromaRdd, ["id", "chroma"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
chromaVec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
#chromaDf.first()
df_vec = chromaDf.select(chromaDf["id"],list_to_vector_udf(chromaDf["chroma"]).alias("chroma"))
#df_vec.first()

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

