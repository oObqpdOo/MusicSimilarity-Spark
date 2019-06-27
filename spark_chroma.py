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

chroma = sc.textFile("features/out1.chroma")
chroma = chroma.map(lambda x: x.split(''))

chroma.first()

kv_chroma= chroma.map(lambda x: (x[0], list(x[1:])))

get_neighbors_chroma("HURRICANE1.mp3")
