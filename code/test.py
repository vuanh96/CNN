import pandas as pd
from datetime import datetime
#
df = pd.DataFrame([(1,2,3),(4,5,6),(7,8,9)], columns=['A','B','C'])


# from datetime import datetime
# df = pd.DataFrame([(1,2,874965758),(4,5,874965358),(7,8,874934758)], columns=['A','B','C'])
# df['D'] = df.apply(lambda row: datetime.fromtimestamp(row['C']), axis=1)
# print(df.groupby(lambda x: df['D'][x].month, axis=0))


# print(df.groupby(lambda idx: df.loc[idx]['A']%2).groups)

a = dict()
b = {1:'a', 2:'b'}
print({}.update(b))