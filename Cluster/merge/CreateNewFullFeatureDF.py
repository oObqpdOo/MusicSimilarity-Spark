import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, Row
from pyspark.sql.functions import * # col, array, lit, udf, min, max, round # lit is used for applying one scalar to every row in a whole column when using withColumn and creating a new column
from pyspark.sql.functions import udf
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F


#if executed in PySpark Shell with pyspark --num-executors 27
#sc.stop()

# Create a SparkConf object
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb \
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb \

conf = SparkConf().setAppName("MergeDatasets").set("yarn.nodemanager.resource.detect-hardware-capabilities" , "True") \
                                              .set("yarn.nodemanager.resource.memory-mb", "196608") \
                                              .set("spark.executor.memory", "20g") \
                                              .set("spark.driver.memory", "20g") \
                                              .set("spark.driver.cores", "4") \
                                              .set("spark.executor.cores", "4") \
                                              .set("spark.executor.instances", "100") \
											  .set("spark.dynamicAllocation.enabled", "True") \
                                              .set("spark.dynamicAllocation.initialExecutors", "100") \
                                              .set("spark.dynamicAllocation.executorIdleTimeout", "30s") \
                                              .set("spark.dynamicAllocation.minExecutors", "100") \
                                              .set("spark.dynamicAllocation.maxExecutors", "100")
                                         
# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sqlContext= SQLContext(sc)


fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("NonDuplicates.parquet").persist()
fullFeatureDF.printSchema()
print(fullFeatureDF.count()) #4644356 -> 5967970-1323614 = 4644356
fullFeatureDF.write.mode("overwrite").save("PreProcessedFull.parquet")
test = fullFeatureDF.dropDuplicates(['id'])
print(test.count()) #4644356

#================================================================================
# 100k
#================================================================================

strippedFeatureDF = fullFeatureDF.limit(100000).persist()
strippedFeatureDF.count()
strippedFeatureDF.write.mode("overwrite").save("PreProcessed100000.parquet")
selection = strippedFeatureDF.select(['id'])
print(selection.take(2))
selection.write.csv("Tracklist100000.csv")

#================================================================================
# 500k
#================================================================================

strippedFeatureDF = fullFeatureDF.limit(500000).persist()
strippedFeatureDF.count()
strippedFeatureDF.write.mode("overwrite").save("PreProcessed500000.parquet")
selection = strippedFeatureDF.select(['id'])
print(selection.take(2))
selection.write.csv("Tracklist500000.csv")

#================================================================================
# 1M
#================================================================================

strippedFeatureDF = fullFeatureDF.limit(1000000).persist()
strippedFeatureDF.count()
strippedFeatureDF.write.mode("overwrite").save("PreProcessed1000000.parquet")
selection = strippedFeatureDF.select(['id'])
print(selection.take(2))
selection.write.csv("Tracklist1000000.csv")

#================================================================================
# 3M
#================================================================================

strippedFeatureDF = fullFeatureDF.limit(3000000).persist()
strippedFeatureDF.count()
strippedFeatureDF.write.mode("overwrite").save("PreProcessed3000000.parquet")
selection = strippedFeatureDF.select(['id'])
print(selection.take(2))
selection.write.csv("Tracklist3000000.csv")

#================================================================================
# Single Artist wont work - no more Metadata atm
#================================================================================
#artist = selection.filter(selection.artist == 'In Flames')



