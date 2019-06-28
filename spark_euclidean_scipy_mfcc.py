import pyspark
import pyspark.ml.feature
import pyspark.ml.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from scipy.spatial import distance
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np


#mfcc = sc.textFile("features/out0.mfcc")

mfcc = sc.textFile("features/out[0-9]*.mfcc")
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0], list(x[1:])))

song = "music/PUNISH2.mp3"

meanRdd = mfcc.map(lambda x: (x[0],(x[1][0].replace(' ', '').replace('[', '').replace(']', '').split(','))))
meanDf = spark.createDataFrame(meanRdd, ["id", "mean"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
meanVec = meanDf.select(meanDf["id"],list_to_vector_udf(meanDf["mean"]).alias("mean"))
#meanVec.first()

varRdd = mfcc.map(lambda x: (x[0],(x[1][1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
varDf = spark.createDataFrame(varRdd, ["id", "var"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
varVec = varDf.select(varDf["id"],list_to_vector_udf(varDf["var"]).alias("var"))
#varVec.first()

covRdd = mfcc.map(lambda x: (x[0],(x[1][2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
covDf = spark.createDataFrame(covRdd, ["id", "cov"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
covVec = covDf.select(covDf["id"],list_to_vector_udf(covDf["cov"]).alias("cov"))
#covVec.first()

#mfccRdd = meanRdd.cogroup(varRdd)
#mfccRdd = mfccRdd.cogroup(covRdd)

mfccDf = meanVec.join(varVec, on=['id'], how='left_outer')
mfccDf = mfccDf.join(covVec, on=['id'], how='left_outer')
#mfccDf.first()

assembler = VectorAssembler(inputCols=["mean", "var", "cov"],outputCol="features")

mfccDfMerged = assembler.transform(mfccDf)
#print("Assembled columns 'mean', 'var', 'cov' to vector column 'features'")
#mfccDfMerged.select("features", "id").show(truncate=False)
#mfccDfMerged.first()

df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
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
    

