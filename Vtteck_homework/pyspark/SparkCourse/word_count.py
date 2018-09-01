from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

lines = spark.read.text('word_count.txt').rdd.map(lambda r: r[0])
counts = lines.flatMap(lambda x: x.split(' ')) \
                .map(lambda x: (x, 1)) \
                .reduceByKey(lambda a, b: a+b)
output = counts.collect()
for (word, count) in output:
    print("%s: %i" % (word, count))

spark.stop()