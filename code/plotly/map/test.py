import pandas as pd
from datetime import datetime
#
# df = pd.DataFrame([(1,2,3),(4,5,6),(7,8,9)], columns=['A','B','C'])
# print(df[['A', 'C']].values)

# from datetime import datetime
# df = pd.DataFrame([(1,2,874965758),(4,5,874965358),(7,8,874934758)], columns=['A','B','C'])
# df['D'] = df.apply(lambda row: datetime.fromtimestamp(row['C']), axis=1)
# print(df.groupby(lambda x: df['D'][x].month, axis=0))

# Loading data from file csv
users = pd.read_csv('ml-100k/u.user.zipcode', header=0)
columns_used = ['USER ID', 'AGE', 'GENDER', 'OCCUPATION', 'ZIP CODE', 'POST OFFICE CITY']

items = pd.read_csv('ml-100k/u.item', sep='|', header=None)

ratings = pd.read_csv('ml-100k/ub.base', sep='\t', header=None, names=['USER ID', 'MOVIE ID', 'RATING', 'TIME STAMP'])

def group_ratings_by_month(df, idx, col):
    time = datetime.fromtimestamp(df.loc[idx][col])
    return time.year, time.month

ratings_by_user = ratings[ratings['USER ID'] == 1]
ratings_by_month = ratings_by_user.groupby('RATING')\
                                .count().reset_index()
for i in ratings_by_month['RATING']:
    print(i)