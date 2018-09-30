from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseVector
from scipy.sparse import csr_matrix, vstack, save_npz
import numpy as np
import time

s = time.time()
# Initialize spark session
spark = SparkSession \
    .builder \
    .appName("read_csv") \
    .getOrCreate()

schema = StructType([
    StructField("ID_USER", StringType()),
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

data_dir = 'datasource'

# Read csv file to dataframe
df = spark.read.csv(data_dir + '/' + '*.csv', sep=',', header=True, schema=schema)
df.show()

# Determine max values of VolData, NoSMS, DurationVoice
# option 1:
max_values = df.groupBy().max('VolData', 'NoSMS', 'DurationVoice', 'WEEK').collect()[0]
n_weeks = max_values[3] + 1

# option 2:
# df.createOrReplaceTempView('user')
# max_values = spark.sql('SELECT MAX(VolData), MAX(NoSMS), MAX(DurationVoice) FROM user').collect()[0]

print(max_values)

# Min-max value scale of VolData, NoSMS, DurationVoice
df = df.withColumn('VolData', df['VolData'] / max_values[0]) \
    .withColumn('NoSMS', df['NoSMS'] / max_values[1]) \
    .withColumn('DurationVoice', df['DurationVoice'] / max_values[2])
df.show()

# Create features formatted to vector[504]
n_features = 3  # number of features used


def parse(x):
    """
    map each record x according to format with
    key = ('ID_USER','WEEK) and
    value = [(idx1, 'VolData'), (idx2, 'NoSMS'), (idx3, 'DurationVoice')]
    - idx* is corresponding index in vector[504] to create sparse vector
    :param x:
    :return: (key, value)
    """
    key = (x['ID_USER'], x['WEEK'])
    value = []
    if x['VolData'] > 0:
        value += [(24 * n_features * x['DoW'] + n_features * x['HOUR'], x['VolData'])]
    if x['NoSMS'] > 0:
        value += [(24 * n_features * x['DoW'] + n_features * x['HOUR'] + 1, x['NoSMS'])]
    if x['DurationVoice'] > 0:
        value += [(24 * n_features * x['DoW'] + n_features * x['HOUR'] + 2, x['DurationVoice'])]
    return key, value


# map each record x to (key, value) with function parse()
rdd = df.rdd.map(lambda x: parse(x))
# concatenate values (concatenate list of (idx, val)) to sparse vector
rdd = rdd.reduceByKey(lambda a, b: a + b)
# create features is sparse vector
rdd = rdd.map(lambda x: (x[0][0], x[0][1], SparseVector(24 * 7 * n_features, x[1])))

# write features to csv file by week
df = spark.createDataFrame(rdd, ['ID_USER', 'WEEK', 'FEATURES'])
df.show()


def as_matrix(vec):
    data, indices = vec.values, vec.indices
    shape = 1, vec.size
    return csr_matrix((data, indices, np.array([0, data.size])), shape)


for week in [1]:
    features = df.filter(df['WEEK'] == week).rdd.map(lambda x: x['FEATURES'])
    mats = features.map(as_matrix)
    mat = mats.reduce(lambda x, y: vstack([x, y]))
    save_npz('data/preprocessed/inputs_week' + str(week) + '.npz', mat)


spark.stop()

e = time.time()
print("Time: %f s" % (e - s))
