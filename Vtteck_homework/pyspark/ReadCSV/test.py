from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseVector, SparseMatrix
from scipy.sparse import csr_matrix, vstack, save_npz
from operator import attrgetter
import numpy as np
from pyspark.sql.functions import stddev, max
import time

s = time.time()
# Initialize spark session
spark = SparkSession \
    .builder \
    .appName("test") \
    .getOrCreate()

mat1 = SparseMatrix(2, 2, [0, 2, 3], [0, 1, 1], [2, 3, 4], True)
print(mat1.toArray())
mat2 = SparseMatrix(1, 2, [0, 1], [1], [5], True)
print(mat2.toArray())
mat3 = SparseMatrix(1, 2, [0, 2], [0, 1], [6, 7], True)
print(mat3.toArray())

def vstack_pyspark(mats):
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


mat = vstack_pyspark([mat3, mat2])
print(vstack_pyspark([mat1, mat]).toArray())



spark.stop()

e = time.time()
print("Time: %f s" % (e - s))
