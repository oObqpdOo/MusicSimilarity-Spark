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

#rhythms = sc.textFile("out.rh.h5")

#===> Read Rhythm Histogram
rhythms = sc.textFile("out.rh")
print rhythms.top(2)

#===> Split Data per ","
#rdd = sc.parallelize(rhythms)
rhythms = rhythms.map(lambda x: x.split(","))
print rhythms.top(2)

#===> Create Key-Value Pairs of Trackname - RH-Feature
kv_rhythm = rhythms.map(lambda x: (x[0], list(x[1:])))
print kv_rhythm.top(2)

#===> create a dataframe
#df = kv_rhythm.toDF(["id","features"])


def get_neighbors(song):
    #===> get distinct element
    comparator = kv_rhythm.lookup(song)
    #print comparator

    #===> get key of distince element
    comparator_value = comparator[0]
    #print comparator_value

    #===> get first element
    #comparator = kv_rhythm.first()
    #print comparator

    #===> get key of first element
    #comparator_key = comparator[0]
    #comparator_value = comparator[1]
    #print comparator_key

    df = spark.createDataFrame(kv_rhythm, ["id", "features"])

    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_vec = df.select(
        df["id"], 
        list_to_vector_udf(df["features"]).alias("features")
    )

    brp = BucketedRandomProjectionLSH(
        inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0
    )
    model = brp.fit(df_vec)

    #one_row = df_vec.first().features
    #print(df_vec.first().id)

    #===> get key of distince element
    comparator_value = Vectors.dense(comparator[0])
    #print comparator_value

    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result

    #https://stackoverflow.com/questions/46725290/pyspark-euclidean-distance-between-entry-and-column
    #result = model.approxSimilarityJoin(df_vec, df_vec, 100.0, #distCol="EuclideanDistance")
    #print result.collect()




get_neighbors("HURRICANE1.mp3")
print result
