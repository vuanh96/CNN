from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import max
import time

# ------------------Compare performance some ways to find max of columns in pyspark-------------------------------

# Initialize spark session
spark = SparkSession \
    .builder \
    .appName("test_performance") \
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
rdd = df.rdd.map(lambda x: x)
df.show()

# ---------------------DataFrame1------------------------
s = time.time()
# Determine max values of VolData, NoSMS, DurationVoice
max_values = df.groupBy().max('VolData', 'NoSMS', 'DurationVoice', 'WEEK').collect()[0]
print(max_values)

e1 = time.time()
print("DF1 - Time: %f s" % (e1 - s))
# ---------------------DataFrame2------------------------
# Determine max values of VolData, NoSMS, DurationVoice
max_values = []
for feature in ['VolData', 'NoSMS', 'DurationVoice']:
    max_values.append(df.groupBy().max(feature).collect()[0][0])
print(max_values)

e2 = time.time()
print("DF2 - Time: %f s" % (e2 - e1))

# ---------------------DataFrame3--------------------------
# Determine max values of VolData, NoSMS, DurationVoice
max_values = df.select(max('VolData'), max('NoSMS'), max('DurationVoice')).collect()[0]
print(max_values)

e3 = time.time()
print("DF3 - Time: %f s" % (e3 - e2))
# ---------------------DataFrame4--------------------------
# Determine max values of VolData, NoSMS, DurationVoice
max_values = df.groupBy().agg(max('VolData'), max('NoSMS'), max('DurationVoice')).collect()[0]
print(max_values)

e4 = time.time()
print("DF4 - Time: %f s" % (e4 - e3))

# *******NOTE********: DF3 and DF4 can calculate functions in pyspark.sql.functions, example: stddev, log2, md5...

# -----------------------RDD------------------------------
max_values = []
for feature in ['VolData', 'NoSMS', 'DurationVoice']:
    max_values.append(rdd.max(key=lambda x: x[feature])[feature])
n_weeks = rdd.max(key=lambda x: x['WEEK'])['WEEK']
print(max_values)

e5 = time.time()
print("RDD - Time: %f s" % (e5 - e4))
# ----------------------SQL--------------------------------
df.createOrReplaceTempView('user')
max_values = spark.sql('SELECT MAX(VolData), MAX(NoSMS), MAX(DurationVoice) FROM user').collect()[0]
print(max_values)

e6 = time.time()
print("SQL - Time: %f s" % (e6 - e5))


spark.stop()
