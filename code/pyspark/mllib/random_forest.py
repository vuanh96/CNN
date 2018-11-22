from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("Random Forest")\
    .getOrCreate()


# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
# data.show()

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                            numTrees=100, maxDepth=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
# predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
# accuracy = evaluator.evaluate(predictions)
# print("Accuracy = %f" % (1.0-accuracy))

# Compute raw scores on the test set
predictionAndLabels = predictions.rdd.map(lambda row: (float(row["predictedLabel"]), row["label"]))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Statistics by class
labels = data.select("label").distinct().collect()
print("%10s %10s %10s %10s" % ("Class", "Precision", "Recall", "F1"))
for label in sorted(labels):
    label = label[0]
    print("%10s %10.3f %10.3f %10.3f" % (label, metrics.precision(label), metrics.recall(label), metrics.fMeasure(label, beta=1.0)))

