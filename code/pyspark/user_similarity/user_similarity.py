from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix, MatrixEntry
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import sqrt
from scipy.sparse import load_npz, coo_matrix
import time


def compute_cosine_similarity(row):
    features1 = row[0][1]
    features2 = row[1][1]
    sum_xx = sum_yy = sum_xy = 0
    for i in range(len(features1)):
        sum_xy += features1[i] * features2[i]
        sum_xx += features1[i] * features1[i]
        sum_yy += features2[i] * features2[i]

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if denominator:
        score = (numerator / (float(denominator)))

    return score
    # return cosine_similarity(np.array(row[0][1]).reshape(1,-1), np.array(row[1][1]).reshape(1,-1))[0][0]


s = time.time()
spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName("nbcf") \
        .getOrCreate()

# Way 1:
# rdd = spark.sparkContext.textFile("user_profiles/*.csv")
# # mapping to user => [feature1, feature2,...]
# user_features = rdd.map(lambda r: r.split(',')).map(lambda r: (int(r[0]), [int(val) for val in r[1:]]))
# # pair: (user1, features1),(user2, features2)
# user_pairs = user_features.cartesian(user_features).filter(lambda r: r[0][0] < r[1][0])
# # (user1, user2), similarity
# user_similarities = user_pairs.map(lambda row: (row[0][0], row[1][0], compute_cosine_similarity(row)))
# user_similarities = user_similarities.filter(lambda row: row[2] != 0)  # filter pairs have similarity = 0
#
# print(user_similarities.take(5))
#
# # write user similarities
# user_similarities.saveAsTextFile("user_similarities_calculated_way1")
#
# # load user similarities
# user_similarities_load = spark.sparkContext.textFile("user_similarities_calculated_way1")
# print(user_similarities_load.take(5))

# Way 2:
sparse_mat = load_npz('user_profiles/user_profiles.npz').tocoo()
rows, cols, data = sparse_mat.row, sparse_mat.col, sparse_mat.data
entries = spark.sparkContext.parallelize([MatrixEntry(rows[i], cols[i], data[i]) for i in range(len(data))])
# Convert to CoordinateMatrix => RowMatrix
mat = CoordinateMatrix(entries).transpose().toRowMatrix().persist()

# Calculate exact and approximate similarities
user_similarities = mat.columnSimilarities().entries.map(lambda r: ((r.i, r.j), r.value))

# Output
# print("Have {} pairs user similarity: {} ...".format(user_similarities.count(), user_similarities.take(5)))

# write user similarities
user_similarities.sortByKey().saveAsTextFile("user_similarities_calculated_way2")

# load user similarities
# user_similarities_load = spark.sparkContext.textFile("user_similarities_calculated_way2")
# print(user_similarities_load.take(5))


spark.stop()
e = time.time()
print("Time: %f s" % (e-s))
