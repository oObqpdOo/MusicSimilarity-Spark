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

#mfcc = sc.textFile("features/out0.mfcc")

mfcc = sc.textFile("features/out[0-9]*.mfcc")
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0], list(x[1:])))


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

def get_neighbors_mfcc(song):
    filterDF = mfccDfMerged.filter(mfccDfMerged.id == song)
    filterDF.first()
    comparator_value = filterDF.groupBy("features").mean().collect()[0] 
    comparator_value = Vectors.dense(comparator_value)
    df_vec = mfccDfMerged.select(mfccDfMerged["id"],list_to_vector_udf(mfccDfMerged["features"]).alias("features"))
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
    model = brp.fit(df_vec)
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result   
    result = model.approxSimilarityJoin(df_vec, df_vec, np.inf, distCol="EuclideanDistance")
    print result.collect()
    
song = "music/PUNISH2.mp3"
get_neighbors_mfcc(song)
