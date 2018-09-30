from pyspark.sql import SparkSession
from pyspark.sql.types import *
import time

s = time.time()
spark = SparkSession\
    .builder\
    .appName("read_csv")\
    .getOrCreate()

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

data_dir = 'datasource'
df = spark.read.csv(data_dir + '/' + 'All_User*.csv', sep=',', schema=schema, header=True)
print(df.count())
# df.show()
# df.describe().show()

# Option 1:
# df.createOrReplaceTempView('user')
# all_users = spark.sql('SELECT DISTINCT id_user FROM user')
# users_active_april = spark.sql('SELECT DISTINCT id_user FROM user WHERE id_day BETWEEN 19 AND 21')

# Option 2:
all_users = df.select('ID_USER').distinct()
users_active_april = df.filter('ID_DAY >= 19 AND ID_DAY <=21').select('ID_USER').distinct()

users_inactive_april = all_users.subtract(users_active_april)
print("Number of users inactive on April: ", users_inactive_april.count())
users_inactive_april.show()
#
users_inactive_april.toPandas().to_csv('churn.csv', index=False)
users_active_april.toPandas().head(500000).to_csv('unchurn.csv', index=False)

spark.stop()

e = time.time()
print("Time: %f s" % (e-s))
