from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseVector, SparseMatrix
from scipy.sparse import csr_matrix, vstack, save_npz
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
        value = [(idx1, 'VolData'), (idx2, 'NoSMS'), (idx3, 'DurationVoice')...]
        - idx* is corresponding index in vector[504] to create sparse vector
        :param row
        :return: (key, value)
        """
        key = (row['ID_USER'], row['WEEK'])
        value = []
        for i, feature in enumerate(self.features):
            if row[feature] > 0:
                value += [(24 * self.n_features * row['DoW'] + self.n_features * row['HOUR'] + i, row[feature])]
        return key, value

    def to_csr_matrix(self, vec):
        """
        convert spark sparse vectors to scipy csr_matrix
        """
        data, indices = vec.values, vec.indices
        shape = 1, vec.size
        return csr_matrix((data, indices, [0, data.size]), shape)

    def to_csr_matrix_pyspark(self, vec):
        """
        convert spark sparse vectors to spark csr_matrix
        """
        data, indices = vec.values, vec.indices
        return SparseMatrix(1, vec.size, [0, data.size], indices, data, True)

    def vstack_pyspark(self, mats):
        """
        Stack sparse matrices vertically
        """
        n_rows = mats[0].numRows
        n_cols = mats[0].numCols
        row_ptrs = mats[0].colPtrs
        col_indices = mats[0].rowIndices
        values = mats[0].values
        for mat in mats[1:]:
            n_rows += mat.numRows
            row_ptrs = np.concatenate((row_ptrs, mat.colPtrs[1:] + row_ptrs.size))
            col_indices = np.concatenate((col_indices, mat.rowIndices))
            values = np.concatenate((values, mat.values))
        return SparseMatrix(n_rows, n_cols, row_ptrs, col_indices, values, True)

    def run(self):
        # 1. Initialize spark session
        spark = SparkSession \
            .builder \
            .appName("read_csv") \
            .getOrCreate()

        # 2. Read csv file to dataframe
        df = spark.read.csv(self.data_dir + '/' + '*.csv', sep=',', header=self.header, schema=self.schema)
        df.show()

        # 3. Determine max values of features
        df.createOrReplaceTempView('cdr')
        sql_querry = 'SELECT '
        for feature in self.features:
            sql_querry += 'max(%s), min(%s), ' % (feature, feature)
        sql_querry += 'max(week) FROM cdr '
        aggs = spark.sql(sql_querry).collect()[0]

        print(aggs)
        n_weeks = aggs[-1] + 1

        # 4. Min-max value scale of features
        for i, feature in enumerate(self.features):
            if aggs[2*i] != aggs[2*i+1]:
                df = df.withColumn(feature, (df[feature] - aggs[2*i+1]) / (aggs[2*i] - aggs[2*i+1]))
            else:
                df = df.withColumn(feature, 1)

        df.show()

        # 5. Group by (id_user, week) and create sparse feature vector
        # map each record x to (key, value) with function parse()
        rdd = df.rdd.map(lambda x: self.parse(x))
        # concatenate values (concatenate list of (idx, val)) to sparse vector
        rdd = rdd.reduceByKey(lambda a, b: a + b)
        # create features is sparse vector
        rdd = rdd.map(lambda x: (x[0][0], x[0][1], SparseVector(24 * 7 * self.n_features, x[1])))
        # print rdd
        for row in rdd.take(5):
            print(row)

        # write features to csv file by week
        if not self.use_sparse_matrix_pyspark:
            for week in range(n_weeks):
                features = rdd.filter(lambda x: x[1] == week).map(lambda x: x[2])
                mats = features.map(self.to_csr_matrix)
                mat = mats.reduce(lambda x, y: vstack([x, y]))
                mat = mat.tocsc()
                save_npz(self.output_dir + '/inputs_week' + str(week) + '.npz', mat)
        else:
            features = rdd.map(lambda x: x[2])
            mats = features.map(self.to_csr_matrix_pyspark)
            mat = mats.reduce(lambda x, y: self.vstack_pyspark([x, y]))
            print(mat)

        spark.stop()


if __name__ == '__main__':
    s = time.time()

    schema = StructType([
        StructField("ID_USER", StringType()),
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
    pre = Preprocess(data_dir='datasource', output_dir='data/preprocessed', schema=schema, header=True,
                     features=['RevVoice', 'RevSMS', 'RevData', 'Balance', 'VolData', 'NoSMS', 'DurationVoice'],
                     use_sparse_matrix_pyspark=True)
    pre.run()
    e = time.time()
    print("Time: %f s" % (e - s))
