from pyspark.sql import SparkSession
from pyspark.sql.types import Row, IntegerType, LongType, StructField, StructType, ArrayType, FloatType, DataType, \
    DoubleType, MapType
from pyspark.ml.linalg import SparseMatrix, VectorUDT
from scipy.sparse import csr_matrix, vstack, save_npz
from pyspark.sql.functions import udf, array, sum, max, min, collect_list, pandas_udf, PandasUDFType
import numpy as np
import pandas as pd
from scipy import stats
import time

data_dir = 'data'
output_dir = 'data/preprocessed'
features = ['RevVoice', 'RevSMS', 'RevData', 'Balance', 'VolData', 'NoSMS', 'DurationVoice']
n_features = len(features)

s = time.time()
schema = StructType([
    StructField("ID_USER", LongType()),
    StructField("ID_DAY", IntegerType()),
    StructField("HOUR", IntegerType()),
    StructField("DoW", IntegerType()),
    StructField("WEEK", IntegerType()),
    StructField("RevVoice", IntegerType()),
    StructField("RevSMS", IntegerType()),
    StructField("RevData", IntegerType()),
    StructField("Balance", IntegerType()),
    StructField("VolData", IntegerType()),
    StructField("NoSMS", IntegerType()),
    StructField("DurationVoice", IntegerType())
])
# 1. Initialize spark session
spark = SparkSession \
    .builder \
    .appName("preprocess") \
    .getOrCreate()

# 2. Read csv file to dataframe
df = spark.read.csv(data_dir + '/' + '*.csv', sep=',', header=True, schema=schema).dropna()
df.show()

# 3. Determine max values of features
aggs = df.select([max(c) for c in features] + [min(c) for c in features]).collect()[0]
maxs = aggs[0:n_features]
mins = aggs[n_features:]
print("Max values: ", maxs)
print("Min values: ", mins)

# 4. Min-max value scale of features
for i, feature in enumerate(features):
    if maxs[i] != mins[i]:
        df = df.withColumn(feature, (df[feature] - mins[i]) / (maxs[i] - mins[i]))
    else:
        df = df.withColumn(feature, 1)

df.show()


@udf(StructType([StructField('values', ArrayType(FloatType())),
                 StructField('indices', ArrayType(FloatType()))
                 ]))
def parse(cols):
    indices = []
    values = []
    for i in range(2, n_features + 2):
        if cols[i] != 0:
            values += [cols[i]]
            indices += [24 * n_features * (cols[0] - 1) + n_features * cols[1] + i]
    return Row(values=values, indices=indices)


df = df.withColumn('values-indices', parse(array(['DoW', 'HOUR'] + features)))
df = df.select('ID_USER', 'WEEK', 'values-indices.*')
df.show()

# @pandas_udf(StructType([StructField('ID_USER', LongType()),
#                         StructField('WEEK', IntegerType()),
#                         StructField('values', ArrayType(FloatType())),
#                         StructField('indices', ArrayType(FloatType()))]), PandasUDFType.GROUPED_MAP)
# def concat(pdf):
#     v = pdf.WEEK
#     return pdf.assign(WEEK=(v - v.mean()) / v.std())
#
#
# df = df.groupBy('ID_USER').apply(concat)
# df.show()
rdd = df.rdd.map(lambda x: ((x[0], x[1]),(x[2],x[3])))
rdd = rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
mats = rdd.map(lambda x: (x[0][1], csr_matrix((x[1][0], x[1][1], [0, len(x[1][0])]),
                                              shape=(1, 24 * 7 * n_features))))
mats = mats.reduceByKey(lambda x, y: vstack([x, y]))
print('Time: %f s' % (time.time() - s))


def save_csr_npz(row):
    save_npz(output_dir + '/inputs_week' + str(row[0]) + '.npz', row[1])


mats.foreach(save_csr_npz)

spark.stop()
e = time.time()
print("Time: %f s" % (e - s))
