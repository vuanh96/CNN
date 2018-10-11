from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row, SparkSession
import time

s = time.time()

spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName("ALS") \
        .getOrCreate()


lines = spark.read.text('ml-100k/u.data').rdd
parts = lines.map(lambda row: row.value.split())
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01,
          # implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Save and load model
# model.save('save_model/ALS/' + data_dir)
# model = ALSModel.load('save_model/ALS/ml-100k')

# # Generate top 10 movie recommendations for each use
# userRecs = model.recommendForAllUsers(10)
# # Generate top 10 user recommendations for each movie
# movieRecs = model.recommendForAllItems(10)

spark.stop()

e = time.time()
print("Time: %f s" % (e - s))