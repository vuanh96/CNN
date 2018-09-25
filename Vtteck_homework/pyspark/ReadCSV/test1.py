# from pyspark.sql import SparkSession
# from pyspark.sql.types import *
# from pyspark.ml.linalg import SparseVector, Vectors, VectorUDT, DenseVector
# from scipy.sparse import csr_matrix, vstack, save_npz
# import numpy as np
# from pyspark.sql.functions import udf, sum, array, explode, collect_list, pandas_udf, PandasUDFType, max
# import time
#
# s = time.time()
# # Initialize spark session
# spark = SparkSession \
#     .builder \
#     .appName("read_csv") \
#     .getOrCreate()
#
# rdd = spark.sparkContext.parallelize([(1,2), (1,4), (5, 6)])
# print(rdd.groupByKey().collect())
#
# spark.stop()
#

print(list(range(1,100)))