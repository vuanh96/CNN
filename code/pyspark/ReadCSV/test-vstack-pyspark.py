from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseMatrix
import numpy as np
import time
"""
    DONE!
"""
s = time.time()

spark = SparkSession \
    .builder \
    .appName("test") \
    .getOrCreate()

mat1 = SparseMatrix(2, 3, [0, 2, 4], [0, 1, 0, 2], [1, 2, 3, 6], True)
print('mat1 = ', mat1.toArray())
mat2 = SparseMatrix(2, 3, [0, 1, 3], [2, 0, 2], [9, 10, 12], True)
print('mat2 = ', mat2.toArray())
mat3 = SparseMatrix(1, 3, [0, 1], [2], [1], True)
print('mat3 = ', mat3.toArray())


# Test stack matrices vertically
def vstack_pyspark(mats):
    n_rows = 0
    n_cols = mats[0].numCols
    row_ptrs = [0]
    col_indices = []
    values = []
    for mat in mats:
        n_rows += mat.numRows
        row_ptrs = np.concatenate((row_ptrs, mat.colPtrs[1:] + row_ptrs[-1]))
        col_indices = np.concatenate((col_indices, mat.rowIndices))
        values = np.concatenate((values, mat.values))
    return SparseMatrix(n_rows, n_cols, row_ptrs, col_indices, values, True)


print('mat = ', vstack_pyspark([mat1, mat2, mat3]).toArray())

# Test convert csr_matrix to csc_matrix in pyspark


spark.stop()

e = time.time()
print("Time: %f s" % (e - s))
