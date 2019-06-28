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
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from scipy.spatial import distance

song = "music/PUNISH2.mp3"
rh = sc.textFile("features/out.rh")
rh = rh.map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0], list(x[1:])))

#def euclidean(x, y): 
#    distance = 0 
#    math.sqrt(sum( (a - b)**2 for a, b in zip(a, b)))
#    return float(math.sqrt(distance))

comparator = kv_rh.lookup(song)
comparator_value = comparator[0]
df = spark.createDataFrame(kv_rh, ["id", "features"])
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df_vec = df.select(df["id"],list_to_vector_udf(df["features"]).alias("features"))
comparator_value = Vectors.dense(comparator[0])
print comparator_value
print df_vec.first()

#squared euclidean distance
#result = df.withColumn('distances', Vectors.sqdist(v1, v2))

distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
result = df_vec.withColumn('distances', distance_udf(F.col('features')))

result = result.select("id", "distances").orderBy('distances', ascending=True)
result = result.rdd.flatMap(list).collect()
print result
