from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import SparkSession
import time

s = time.time()

spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName("ALS_rdd") \
        .getOrCreate()
sc = spark.sparkContext

# Load and parse the data
data = sc.textFile('ml-100k/u.data')
ratings = data.map(lambda l: l.split())\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Build the recommendation model using Alternating Least Squares based on implicit ratings
# model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
# model.save(sc, "save_model_ALS/ALS_rdd/ml-100k")
# sameModel = MatrixFactorizationModel.load(sc, "save_model/ALS_rdd/" + data_dir)

e = time.time()
print("Time: %f s" % (e - s))