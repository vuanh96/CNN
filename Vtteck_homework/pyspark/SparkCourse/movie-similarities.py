from pyspark.storagelevel import StorageLevel
from math import sqrt
import time

from pyspark.sql import SparkSession


def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.item", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


def mean(movies_ratings):
    n = 0
    sum = 0
    for movie, rating in movies_ratings:
        n += 1
        sum += rating
    return sum / n


# Python 3 doesn't let you pass around unpacked tuples,
# so we explicitly extract the ratings now.
def makePairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return (movie1, movie2), (rating1, rating2)


def filterDuplicates(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if denominator:
        score = (numerator / (float(denominator)))

    return score, numPairs


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .appName("MovieSimilarities") \
        .getOrCreate()

    print("\nLoading movie names...")
    nameDict = loadMovieNames()
    print(len(nameDict))

    data = spark.sparkContext.textFile("ml-100k/u.data")

    # Map ratings to key / value pairs: user ID => movie ID, rating
    ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

    # Determine mean of ratings by user
    means = ratings.groupByKey().mapValues(mean)
    print(means.take(5))

    # Normalize rating by subtract mean rating
    ratings = ratings.join(means).map(lambda l: (l[0], (l[1][0][0], l[1][0][1] - l[1][1])))

    # Emit every movie rated together by the same user.
    # Self-join to find every combination.
    joinedRatings = ratings.join(ratings)

    # At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

    # Filter out duplicate pairs
    uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

    # Now key by (movie1, movie2) pairs.
    moviePairs = uniqueJoinedRatings.map(makePairs)

    # We now have (movie1, movie2) => (rating1, rating2)
    # Now collect all ratings for each movie pair and compute similarity
    moviePairRatings = moviePairs.groupByKey()

    # We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
    # Can now compute similarities.
    moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity)

    # Save the results if desired
    # moviePairSimilarities.sortByKey()
    # moviePairSimilarities.saveAsTextFile("movie-sims")

    # Extract similarities for the movie we care about that are "good".
    scoreThreshold = 0.97
    coOccurrenceThreshold = 50

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    s = time.time()
    for i in range(1, 2):
        movieID = i
        filteredResults = moviePairSimilarities.filter(
            lambda pairSim: (pairSim[0][0] == movieID or pairSim[0][1] == movieID)
                            and pairSim[1][0] > scoreThreshold
                            and pairSim[1][1] > coOccurrenceThreshold)

        # Sort by quality score.
        results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending=False).take(10)

        # print("Top 10 similar movies for " + nameDict[movieID])
        # for result in results:
        #     (sim, pair) = result
        #     # Display the similarity result that isn't the movie we're looking at
        #     similarMovieID = pair[0]
        #     if similarMovieID == movieID:
        #         similarMovieID = pair[1]
        #     print(str(similarMovieID) + ' : ' + nameDict[similarMovieID] + "\tscore: " + str(sim[0])
        #           + "\tstrength: " + str(sim[1]))

    spark.stop()
    e = time.time()
    print("Time: %f s" % (e - s))

# Input: 1682 movies, 100,000 ratings
# Result:
# +----------------------------+------+------+------+-----+
# |Number of movies            |     1|    10|    20|     |
# +----------------------------+------+------+------+-----+
# |persist(MEMORY_ONLY)=cache()|    88|   217|   353|     |
# |persist(DISK_ONLY)          |    88|   218|   356|     |
# |no persist                  |   100|   485|  1355|     |
# +----------------------------+------+------+------+-----+
