from uszipcode import SearchEngine
import pandas as pd

search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database


def get_location_by_zipcode(row):
    loc = search.by_zipcode(row['ZIP CODE'])
    return pd.Series([loc.lat, loc.lng, loc.post_office_city])


user = pd.read_csv("ml-100k/u.user", sep='|', header=None, names=['USER ID', 'AGE', 'GENDER', 'OCCUPATION', 'ZIP CODE'])
user[['LAT', 'LON', 'POST OFFICE CITY']] = user.apply(get_location_by_zipcode, axis=1)
print(user)
user.to_csv('ml-100k/u.user.zipcode', index=False)