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

chroma = sc.textFile("out.chroma")
chroma = chroma.map(lambda x: x.split(','))

chroma.first()

kv_chroma= chroma.map(lambda x: (x[0], list(x[1:])))

def get_neighbors_chroma(song):
    comparator = kv_chroma.lookup(song)
    comparator_value = comparator[0]
    df = spark.createDataFrame(kv_chroma, ["id", "features"])
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
    model = brp.fit(df_vec)
    comparator_value = Vectors.dense(comparator[0])
    result = model.approxNearestNeighbors(df_vec, comparator_value, df_vec.count()).collect()
    rf = spark.createDataFrame(result)
    result = rf.select("id", "distCol").rdd.flatMap(list).collect()
    print result


get_neighbors_chroma("HURRICANE1.mp3")
