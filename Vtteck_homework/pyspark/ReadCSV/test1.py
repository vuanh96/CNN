# from pyspark.sql import SparkSession
# from pyspark.sql.types import Row, StructField, StructType, IntegerType, ArrayType
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
#
# df = spark.createDataFrame([(1,2,3), (4,5,6), (7,8,9)], ['id', 'v1', 'v2'])
# df.show()
# cols = ['id', 'v1', 'v2']
# maxs = df.groupBy().max(*cols).collect[0]
# print(maxs)
#
# spark.stop()
#
# e = time.time()
# print("Time: %f s" % (e - s))
#
# # cols = ('1','2','3')
# # print(*cols)

import pandas as pd

df = pd.DataFrame([(1,[1,2,3]), (1,[4,5,6])], columns=['id','v'])
df = df.groupby('id')['v'].sum().reset_index()


print(df)