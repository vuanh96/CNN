from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseMatrix
from scipy.sparse import csr_matrix, vstack, save_npz
from pyspark.sql.functions import max, min
import numpy as np
import time


class Preprocess:
    def __init__(self, data_dir, output_dir, features, schema=None, header=False, use_sparse_matrix_pyspark=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.features = features
        self.schema = schema
        self.header = header
        self.use_sparse_matrix_pyspark = use_sparse_matrix_pyspark
        self.n_features = len(self.features)

    def parse(self, row):
        """
        map each record x according to format with
        key = ('ID_USER','WEEK) and
        value = (values, indices) corresponding in vector to create sparse vector
        """
        key = (row['ID_USER'], row['WEEK'])
        indices = []
        values = []
        for i, feature in enumerate(self.features):
            if row[feature] != 0:
                values += [row[feature]]
                indices += [24 * self.n_features * (row['DoW']-1) + self.n_features * row['HOUR'] + i]

        return key, (values, indices)

    def to_csr_matrix(self, pairs):
        data = []
        indices = []
        idxptr = [0]
        shape = len(pairs), 24*7*self.n_features
        for vals, idxs in pairs:
            data += vals
            indices += idxs
            idxptr.append(idxptr[-1] + len(vals))

        return csr_matrix((data, indices, idxptr), shape=shape)

    def save_csr_npz(self, row):
        save_npz(self.output_dir + '/inputs_week' + str(row[0]) + '.npz', row[1])

    def to_csr_matrix_pyspark(self, pairs):
        n_rows = len(pairs)
        n_cols = 24*7*self.n_features
        values = []
        col_indices = []
        row_ptrs = [0]
        for vals, idxs in pairs:
            values += vals
            col_indices += idxs
            row_ptrs.append(row_ptrs[-1] + len(vals))

        return SparseMatrix(n_rows, n_cols, row_ptrs, col_indices, values, True)

    def run(self):
        # 1. Initialize spark session
        spark = SparkSession \
            .builder\
            .master('local[*]') \
            .appName('preprocess') \
            .getOrCreate()

        # 2. Read csv file to dataframe
        df = spark.read.csv(self.data_dir + '/' + '*.csv', sep=',', header=self.header, schema=self.schema).dropna()
        df.show()

        # 3. Determine max values of features
        aggs = df.select([max(c) for c in self.features] + [min(c) for c in self.features]).collect()[0]
        maxs = aggs[0:self.n_features]
        mins = aggs[self.n_features:]
        print("Max values: ", maxs)
        print("Min values: ", mins)

        # 4. Min-max value scale of features
        for i, feature in enumerate(self.features):
            if maxs[i] != mins[i]:
                df = df.withColumn(feature, (df[feature] - mins[i]) / (maxs[i] - mins[i]))
            else:
                df = df.withColumn(feature, 1)

        df.show()

        # 5. Group by (id_user, week) and create sparse feature vector
        rdd = df.rdd.map(lambda x: self.parse(x))  # ((id_user, week), (values, indices))
        # concatenate of values and indices by key
        rdd = rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        rdd = rdd.map(lambda x: (x[0][1], x[1]))  # (week, (values, indices))

        # write features to csv file by week
        if not self.use_sparse_matrix_pyspark:
            # rows = (week, csr_matrix_scipy of week)
            # mats = rdd.map(lambda x: (x[0][1], csr_matrix((x[1][0], x[1][1], [0, len(x[1][0])]),
            #                                               shape=(1, 24 * 7 * self.n_features))))
            # mats = mats.reduceByKey(lambda x, y: vstack([x, y]))
            mats = rdd.groupByKey().mapValues(self.to_csr_matrix)
            mats.foreach(self.save_csr_npz)
        else:
            # rows = (week, csr_matrix_pyspark of week)
            mats = rdd.groupByKey().mapValues(self.to_csr_matrix_pyspark)
            mats.foreach(lambda x: print(x[0], ' : ', x[1].toArray()))

        spark.stop()


if __name__ == '__main__':

    s = time.time()

    schema = StructType([
        StructField("ID_USER", LongType()),
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
    pre = Preprocess(data_dir='data', output_dir='data/preprocessed', schema=schema, header=True,
                     features=['RevVoice', 'RevSMS', 'RevData', 'Balance', 'VolData', 'NoSMS', 'DurationVoice'],
                     use_sparse_matrix_pyspark=False)
    pre.run()
    e = time.time()
    print("Time: %f s" % (e - s))
