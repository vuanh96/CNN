from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseMatrix
from scipy.sparse import csr_matrix, vstack, save_npz
import time

"""
    DONE
"""


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
        indices = []
        values = []
        for i, feature in enumerate(self.features):
            if row[feature] != 0:
                values += [row[feature]]
                indices += [2 * self.n_features * (row['DoW'] - 1) + self.n_features * row['HOUR'] + i]

        return key, (values, indices)

    def to_csr_matrix(self, pairs):
        data = []
        indices = []
        idxptr = [0]
        shape = len(pairs), 2 * 2 * self.n_features
        for vals, idxs in pairs:
            data += vals
            indices += idxs
            idxptr.append(idxptr[-1] + len(vals))

        return csr_matrix((data, indices, idxptr), shape=shape)

    def save_csc_npz(self, row):
        save_npz(self.output_dir + '/inputs_week' + str(row[0]) + '.npz', row[1].tocsc())

    def to_csr_matrix_pyspark(self, pairs):
        n_rows = len(pairs)
        n_cols = 2 * 2 * self.n_features
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
            .builder \
            .appName("preprocess") \
            .getOrCreate()

        # 2. Read csv file to dataframe
        df = spark.read.csv(self.data_dir + '/' + '*.csv', sep=',', header=self.header, schema=self.schema).dropna()
        df.show()

        # 3. Determine max values of features
        df.createOrReplaceTempView('preprocess')
        sql_querry = 'SELECT '
        for feature in self.features:
            sql_querry += 'max(%s), min(%s), ' % (feature, feature)
        sql_querry += 'max(week) FROM preprocess '
        aggs = spark.sql(sql_querry).collect()[0]
        max_values = aggs[0:-1:2]
        min_values = aggs[1:-1:2]
        print("Max values: ", max_values)
        print("Min values: ", min_values)

        # 4. Min-max value scale of features
        for i, feature in enumerate(self.features):
            if max_values[i] != min_values[i]:
                df = df.withColumn(feature, (df[feature] - min_values[i]) / (max_values[i] - min_values[i]))
            else:
                df = df.withColumn(feature, 1)

        df.show()

        # 5. Group by (id_user, week) and create sparse feature vector
        # map each record x to (key, (values, indices) with function parse()
        rdd = df.rdd.map(lambda x: self.parse(x))
        # concatenate values (concatenate list of values and indices) to sparse vector
        rdd = rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        rdd = rdd.map(lambda x: (x[0][1], x[1]))  # (week, (values, indices))

        # csr_matrix_scipy
        mats = rdd.groupByKey().mapValues(self.to_csr_matrix)
        mats.foreach(lambda x: print(x[0], ' : ', x[1].toarray()))

        # csr_matrix_pyspark
        mats = rdd.groupByKey().mapValues(self.to_csr_matrix_pyspark)
        mats.foreach(lambda x: print(x[0], ' : ', x[1].toArray()))

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
    pre = Preprocess(data_dir='data', output_dir='', schema=schema, header=True,
                     features=['DurationVoice'])
    pre.run()

    e = time.time()
    print("Time: %f s" % (e - s))
