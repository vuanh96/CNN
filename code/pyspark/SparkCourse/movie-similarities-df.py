from pyspark.sql import SparkSession
from pyspark.sql.types import Row, IntegerType, LongType, StructField, StructType, ArrayType, FloatType, DataType, \
    DoubleType, MapType
from pyspark.ml.linalg import SparseMatrix, VectorUDT
from scipy.sparse import csr_matrix, vstack, save_npz
from pyspark.sql.functions import udf, array, sum, max, min, collect_list, pandas_udf, PandasUDFType, col
import numpy as np
import pandas as pd
from scipy import stats
import time

s = time.time()
schema = StructType([
    StructField("user", IntegerType()),
    StructField("movie", IntegerType()),
    StructField("rating", IntegerType()),
    StructField("time_stamp", LongType()),
])
# 1. Initialize spark session
spark = SparkSession \
    .builder \
    .appName("MovieSimilarities") \
    .getOrCreate()

df = spark.read.csv('ml-100k/u.data', sep='\t', schema=schema)
df = df.select('user', 'movie', 'rating')
df = df.alias('df1').join(df.alias('df2'), 'user', 'inner').select(col('df1.movie').alias('movie1'), col('df2.movie').alias('movie2'),
                                                      col('df1.rating').alias('rating1'), col('df2.rating').alias('rating2'))
df = df.filter(df['movie1'] < df['movie2'])
df.show()
# df = df.filter()



# df.show()
spark.stop()
e = time.time()
print("Time: %f s" % (e - s))
