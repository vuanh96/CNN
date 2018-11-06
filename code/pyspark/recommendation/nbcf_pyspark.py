from math import sqrt
import time
import pandas as pd
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import SparkSession
import numpy as np


def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.item", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


if __name__ == '__main__':

    s = time.time()

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName("NBCF_Pyspark") \
        .getOrCreate()

    # Training:
    # Load training data
    train = spark.sparkContext.textFile("ml-100k/ub.base")

    # movieID => userID, rating
    train = train.map(lambda r: r.split()).map(lambda r: (int(r[1]), (int(r[0]), float(r[2]))))

    # # Determine mean of ratings by user
    # # movieID => mean
    # def mean(val):
    #     s = n = 0
    #     for user, rating in val:
    #         s += rating
    #         n += 1
    #     if n > 0:
    #         return s / n
    #     return 0
    #
    # # means = train.groupByKey().mapValues(mean)
    # sepOp = lambda x, y: (x[0]+y[1], x[1]+1)
    # combOp = lambda x, y: (x[0]+y[0], x[1]+y[1])
    # means = train.aggregateByKey((0, 0), sepOp, combOp).mapValues(lambda v: np.float64(v[0]/v[1]) if v[1] > 0 else 0)
    # # means.sortByKey().map(lambda r: "{} {}".format(r[0], r[1])).saveAsTextFile('mean')
    means = spark.sparkContext.textFile("mean").map(lambda r: r.split()).map(lambda r: (int(r[0]), float(r[1])))

    # userID => (movieID, rating_normalized)
    train = train.join(means).map(lambda r: (r[1][0][0], (r[0], r[1][0][1] - r[1][1]))).cache()

    # # # userID => (movieID, rating)
    # # train = train.map(lambda r: (r[1][0], (r[0], r[1][1])))
    # # # userID => ((movie1, rating1), (movie2, rating2))
    # # joinedRatings = train.join(train)
    # #
    # # # Filter out duplicate pairs
    # # def filterDuplicates(userRatings):
    # #     ratings = userRatings[1]
    # #     (movie1, rating1) = ratings[0]
    # #     (movie2, rating2) = ratings[1]
    # #     return movie1 < movie2
    # #
    # # uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
    # #
    # # # (movie1, movie2) = > (rating1, rating2)
    # # def makePairs(userRatings):
    # #     ratings = userRatings[1]
    # #     (movie1, rating1) = ratings[0]
    # #     (movie2, rating2) = ratings[1]
    # #     return (movie1, movie2), (rating1, rating2)
    # #
    # # moviePairs = uniqueJoinedRatings.map(makePairs)
    # #
    # # # (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
    # # moviePairRatings = moviePairs.groupByKey()
    # #
    # # # Compute similarities
    # # # (movie1, movie2) => (similarity, numUserRated)
    # # def computeCosineSimilarity(ratingPairs):
    # #     numPairs = 0
    # #     sum_xx = sum_yy = sum_xy = 0
    # #     for ratingX, ratingY in ratingPairs:
    # #         sum_xx += ratingX * ratingX
    # #         sum_yy += ratingY * ratingY
    # #         sum_xy += ratingX * ratingY
    # #         numPairs += 1
    # #
    # #     numerator = sum_xy
    # #     denominator = sqrt(sum_xx) * sqrt(sum_yy)
    # #
    # #     score = 0
    # #     if denominator:
    # #         score = (numerator / (float(denominator)))
    # #
    # #     return score, numPairs
    # #
    # # moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()
    #
    # entries = train.map(lambda r: MatrixEntry(r[0], r[1][0], r[1][1]))
    # # Convert to CoordinateMatrix => RowMatrix
    # mat = CoordinateMatrix(entries).toRowMatrix()
    #
    # # Calculate exact and approximate similarities
    # moviePairSimilarities = mat.columnSimilarities().entries.map(lambda r: ((r.i, r.j), r.value)).cache()
    #
    # # Save the results if desired
    # moviePairSimilarities.sortByKey().map(lambda r: "{} {} {}".format(r[0][0], r[0][1], r[1])).saveAsTextFile("movie-sims")
    #
    # Load movie similarity from file
    moviePairSimilarities = spark.sparkContext.textFile("movie-sims")\
        .map(lambda r: r.split()).map(lambda r: ((int(r[0]), int(r[1])), np.float64(r[2])))

    # Evaluate:
    # Load testing data
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    test = pd.read_csv('ml-100k/ub.test.1', sep='\t', names=r_cols).values

    def predict(row):
        userID, movieID, rating_exact = row

        # Step 1: find all movie rated by user
        # (movieID, movieA) => ratingA
        movie_rated_by_userID = train.filter(lambda r: r[0] == userID)\
            .map(lambda r: ((movieID, r[1][0]), r[1][1]) if movieID < r[1][0] else ((r[1][0], movieID), r[1][1]))
        mean_movieID = means.filter(lambda r: r[0] == movieID).first()[1]

        # Step 2: find similarity between the movieID and others which already rated by userID
        # (movieID, movieA) => (ratingA, sim)
        movie_pairs = movie_rated_by_userID.join(moviePairSimilarities)

        # Step 3: Select top k movie which most similar with movieID
        # (ratingA, sim)
        k = 30
        movie_pairs_top = movie_pairs.top(k, key=lambda r: r[1][1])

        rating_pred = movie_pairs_top.map(lambda r: (r[1][0]*r[1][1], abs(r[1][1])))\
            .reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))

        rating_pred = rating_pred[0] / (rating_pred[1] + 1e-8) + mean_movieID

        return (rating_pred - rating_exact)**2

    n_tests = test.shape[0]
    SE = 0  # squared error
    # for i in range(n_tests):
    SE += predict(test[0,:3])
    print("RMSE: ", np.sqrt(SE/n_tests))

    # Load testing data
    test = spark.sparkContext.textFile('ml-100k/ub.test.1')
    # movieY => (userX, ratingXY)
    test = test.map(lambda r: r.split()).map(lambda r: (int(r[1]), (int(r[0]), np.float64(r[2]))))
    # Subtract rating to mean by movie
    # userX => (movieY, ratingXY)
    test = test.join(means).map(lambda r: (r[1][0][0], (r[0], r[1][0][1] - r[1][1]))).cache()
    n_test = test.count()

    # Step 1: Find all movies rated by userX
    # userX => ((movieY, ratingXY), (movieA, ratingXA)) - A is movieID rated by userX
    evaluate = test.join(train)

    # Step 2: Find similarities between movieY and others
    # (movieY, movieA) => (i, userX, ratingXY, ratingXA) - i is index of movieY in key
    def map1(row):
        userX, ((movieY, ratingXY), (movieA, ratingXA)) = row
        if movieY < movieA:
            return (movieY, movieA), (0, userX, ratingXY, ratingXA)
        return (movieA, movieY), (1, userX, ratingXY, ratingXA)

    evaluate = evaluate.map(map1)

    # (movieY, movieA) => ((i, userX, ratingXY, ratingXA), simYA)
    evaluate = evaluate.join(moviePairSimilarities)

    # (userX, movieY, ratingXY) => [(ratingXA, simYA)]
    def map2(row):
        (movieY, movieA), ((i, userX, ratingXY, ratingXA), simYA) = row
        if i == 0:
            return (userX, movieY, ratingXY), [(ratingXA, simYA)]
        return (userX, movieA, ratingXY), [(ratingXA, simYA)]

    evaluate = evaluate.map(map2)

    # (userX, movieY, ratingXY) => [(ratingXA, simYA), (ratingXB, simYB), ...] - A,B is movieID rated by userX
    evaluate = evaluate.reduceByKey(lambda a, b: a + b)

    # Step 3: Take top k movies greatest similarity
    k = 30
    evaluate = evaluate.mapValues(lambda l: sorted(l, key=lambda x: x[1], reverse=True)[:k])

    # Step 4: Predict rating and calculating RMSE
    def predict(row):
        rating_ori = row[0][2]
        val = np.array(row[1])
        ratings = val[:, 0]
        sims = val[:, 1]
        rating_pred = np.dot(ratings, sims) / (np.abs(sims).sum() + 1e-8)
        print((rating_pred - rating_ori) ** 2)
        return (rating_pred - rating_ori) ** 2

    evaluate = evaluate.map(predict).reduce(lambda a, b: a + b)

    # Calculate RMSE
    RMSE = np.sqrt(evaluate / n_test)
    print("RMSE: ", RMSE)

    # # Recommend:
    # # Load movie names
    # print("Loading movie names...")
    # nameDict = loadMovieNames()
    # print("Number of movies: ", len(nameDict))
    # # Extract similarities for the movie we care about that are "good".
    # scoreThreshold = 0.97
    # coOccurrenceThreshold = 50
    #
    # # Filter for movies with this sim that are "good" as defined by
    # # our quality thresholds above
    # for i in range(1, 20):
    #     movieID = i
    #     filteredResults = moviePairSimilarities.filter(
    #         lambda pairSim: (pairSim[0][0] == movieID or pairSim[0][1] == movieID)
    #                         and pairSim[1][0] > scoreThreshold
    #                         and pairSim[1][1] > coOccurrenceThreshold)
    #
    #     # Sort by quality score.
    #     results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending=False).take(10)
    #
    #     print("Top 10 similar movies for " + nameDict[movieID])
    #     for result in results:
    #         (sim, pair) = result
    #         # Display the similarity result that isn't the movie we're looking at
    #         similarMovieID = pair[0]
    #         if similarMovieID == movieID:
    #             similarMovieID = pair[1]
    #         print(str(similarMovieID) + ' : ' + nameDict[similarMovieID] + "\tscore: " + str(sim[0])
    #               + "\tstrength: " + str(sim[1]))

    spark.stop()
    e = time.time()
    print("Time: %f s" % (e - s))

# RMSE:  0.9510678724710572
# Time: 146.198977 s
