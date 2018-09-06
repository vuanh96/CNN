from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
import time


spark = SparkSession\
    .builder\
    .appName("read_csv")\
    .getOrCreate()

df = spark.createDataFrame(((1,1,1),(1,2,2)),['1','2', '3'])
df.show()
rdd = df.rdd.map(lambda x: x[0])
print(rdd.collect())


