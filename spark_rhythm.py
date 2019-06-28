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

rh = sc.textFile("features/out.rh")
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0], list(x[1:])))

rp = sc.textFile("features/out.rp")
rp = rp.map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0], list(x[1:])))

def get_neighbors_rh_brp(song):
    comparator = kv_rh.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_rh, ["id", "features"])
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
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
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
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
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
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
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    comparator_value = Vectors.dense(comparator[0])
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
    result = df_vec.withColumn('distances', distance_udf(F.col('features')))
    result = result.select("id", "distances").orderBy('distances', ascending=True)
    result = result.rdd.flatMap(list).collect()
    print result

song = "music/PUNISH2.mp3"
get_neighbors_rh_brp(song)
get_neighbors_rh_euclidean(song)
get_neighbors_rp_brp(song)
get_neighbors_rp_euclidean(song)
