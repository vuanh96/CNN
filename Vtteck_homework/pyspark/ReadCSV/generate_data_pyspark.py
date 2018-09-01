from pyspark.sql import SparkSession
from pyspark.sql.types import *
import time

s = time.time()
spark = SparkSession\
    .builder\
    .appName("read_csv")\
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

data_dir = 'datasource/'

# read csv file to dataframe
df = spark.read.csv(data_dir + 'All_User*.csv', sep=',', schema=schema, header=True)
df.show()

# determine max values of VolData, NoSMS, DurationVoice
max_values = df.groupBy().max('VolData', 'NoSMS', 'DurationVoice').collect()[0]
print(max_values)

# Min-max value scale of VolData, NoSMS, DurationVoice
df = df.withColumn('VolData', df['VolData']/max_values[0]) \
    .withColumn('NoSMS', df['NoSMS'] / max_values[1]) \
    .withColumn('DurationVoice', df['DurationVoice']/max_values[2])
df.show()



spark.stop()

e = time.time()
print("Time: %f s" % (e-s))
