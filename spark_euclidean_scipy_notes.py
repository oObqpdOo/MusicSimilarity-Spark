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
import org.apache.spark.sql.functions.typedLit
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc

song = "music/TURCA1.wav"

#mfcc = sc.textFile("features/out0.mfcc")

chroma = sc.textFile("features/out[0-9]*.notes")
chroma = chroma.map(lambda x: x.split(';'))
chroma = chroma.map(lambda x: (x[0], x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
chroma = chroma.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
chroma.first()
df = spark.createDataFrame(chroma, ["id", "key", "scale", "notes"])
filterDF = df.filter(df.id == song)
filterDF.first()
comparator_value = filterDF.collect()[0][3] 
print comparator_value
df_merged = df.withColumn("compare", lit(comparator_value))
df_levenshtein = df_merged.withColumn("word1_word2_levenshtein", levenshtein(col("notes"), col("compare")))
df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()

