import pandas as pd
import glob
import numpy as np
# from sklearn.preprocessing import LabelEncoder

data_dir = 'datasource/'
dim = 504  # dimension of input
list_df = []
max_values = [0, 0, 0]  # max values of VolData, NoSMS, DurationVoice
for f in glob.glob(data_dir + '*.csv'):
    df = pd.read_csv(f, sep=',', header=0,
                     names=['ID_USER', 'ID_DAY', 'HOUR', 'DoW', 'WEEK', 'RevVoice', 'RevSMS',
                            'RevData', 'Balance', 'VolData', 'NoSMS', 'DurationVoice']
                     )
    list_df.append(df)

df = pd.concat(list_df, ignore_index=1)
# df['ID_USER'] = LabelEncoder().fit_transform(df['ID_USER'])

for idx, row in df.iterrows():
    sample = np.zeros(dim)



# df.to_csv('data.csv', index=False)
