from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)

FeatureRow = Row('id', 'features')
data = sc.parallelize([(0, Vectors.dense([9.7, 1.0, -3.2])),
                       (1, Vectors.dense([2.25, -11.1, 123.2])),
                       (2, Vectors.dense([-7.2, 1.0, -3.2]))])
df = data.map(lambda r: FeatureRow(*r)).toDF()

vector_udf = udf(lambda vector: float(vector[1]), DoubleType())

df.withColumn('feature_sums', vector_udf(df.features)).first()
