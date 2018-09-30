from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("read_csv")\
    .getOrCreate()

rdd1 = spark.sparkContext.parallelize([[1,2,3,4], [4,5]])
rdd2 = spark.sparkContext.parallelize([("a",["x","y","z"]), ("b",["p", "r"])])
rdd3 = spark.sparkContext.parallelize(range(100))

print(rdd1.reduce(lambda a, b: a+b))