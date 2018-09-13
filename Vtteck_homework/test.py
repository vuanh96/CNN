# from pyspark.sql import SparkSession
# from pyspark.ml.linalg import SparseMatrix
# import numpy as np
# import time
#
# s = time.time()
#
# spark = SparkSession \
#     .builder \
#     .appName("test") \
#     .getOrCreate()
#
# rdd = spark.sparkContext.parallelize([1,2,3,4,5])
# rdd = rdd.flatMap(lambda x: [x, x+1])
# print(rdd.first())
#
# spark.stop()
#
# e = time.time()
# print("Time: %f s" % (e - s))

import numpy as np
a = [0,0,0,0]
a[1] = np.array([1,2,3])
print(a)
