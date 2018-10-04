from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, LongType, StructField, StructType
import time

s = time.time()
schema = StructType([
    StructField("user", IntegerType()),
    StructField("movie", IntegerType()),
    StructField("rating", IntegerType()),
    StructField("time_stamp", LongType())
])
# 1. Initialize spark session
spark = SparkSession \
    .builder \
    .appName("MovieSimilarities") \
    .getOrCreate()

df = spark.read.csv('ml-100k/u.data', sep='\t', ignoreTrailingWhiteSpace=True, inferSchema=True)
# df = df.select('user', 'movie', 'rating')
# df = df.alias('df1').join(df.alias('df2'), 'user', 'inner').select(col('df1.movie').alias('movie1'), col('df2.movie').alias('movie2'),
#                                                       col('df1.rating').alias('rating1'), col('df2.rating').alias('rating2'))
# df = df.filter(df['movie1'] < df['movie2'])

# df = df.filter()



df.show()
spark.stop()
e = time.time()
print("Time: %f s" % (e - s))
