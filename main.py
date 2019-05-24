# Some imports will be used in the notebook.
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest

"""Script for Cleaning data from 
https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurveillanceSystemBRFSS
for the years 2011 - 2017.

There's a little more going on here than is needed for the current blog post, but I'm hoping to continue
to develop this throughout the course.

The dataset is 200mbs in entirety and consists of 464 columns. Many are near duplicate survey questions asked
in different years and are here combined where able.
"""

pd.set_option('display.max_columns', 500)

# Columns copied from the code book. In the file all column names are lower case.

all_columns =[column.lower() for column in ['FRUITJU1', 'FRUIT1', 'FVGREEN', 'VEGETAB1', 'FTJUDA1_', 'FRUTDA1_',
                                            'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN',
                                            '_FRTRESP', '_VEGRESP', '_FRUTSUM', '_VEGESUM', '_FRTLT1', '_VEGLT1',
                                            '_FRUITEX', '_VEGETEX', 'ZIPCODE', 'FRUITJU2', 'FRUIT2', 'FVGREEN1',
                                            'VEGETAB2', 'FRENCHF1', 'POTATOE1', 'FTJUDA2_', 'FRUTDA2_', 'GRENDA1_',
                                            'VEGEDA2_', 'FRNCHDA_', 'POTADA1_', '_MISFRT1', '_MISVEG1', '_FRTRES1',
                                            '_VEGRES1', '_FRUTSU1', '_VEGESU1', '_FRTLT1a', '_VEGLT1a', '_VEGETE1',
                                            'ZIPCODE1', '_FRUITE1', 'YEAR', 'NUMADULT', 'HHADULT', '_AGEG5YR', '_RACE',
                                            'CHILDREN', '_CHLDCNT', 'EMPLOY', 'EMPLOY1', 'INCOME2', '_BMI5CAT',
                                            'GENHLTH', 'PREDIAB1', 'DIABETE3', '_PAINDEX', '_PAINDX1']]

df = pd.read_stata("C:\\Users\Paul\PycharmProjects\BlogPost\data\WA_BRFSS_11to17_B.dta",
                   columns=all_columns, convert_missing=False)

# 77 and 99 are coded as unknown and refused respectively. I'm assuming someone filled out the survey and so score a 1.
df['numadult'] = df['numadult'].map(lambda x: 1 if x in [77, 99] else x)
df['hhadult'] = df['hhadult'].map(lambda x: 1 if x in [77, 99] else x)

df['numadult'] = df['numadult'].fillna(1)
df['hhadult'] = df['hhadult'].fillna(1)

# In case of duplicate entries for the same year, take the larger.
df['adults'] = np.maximum(df['hhadult'], df['numadult'])
df['adults'] = df['adults'].map(lambda x: 1 if x == 0 else x)

# 88 and 99 are coded as unknown and refused respectively. An adult is assumed to be answering and no
# children are assumed.
df['children'] = df['children'].map(lambda x: 0 if x in [88, 99] else x)

df['total_household'] = df['adults'] + df['children']

# Change values for nicer plotting.
df['_race'] = df['_race'].map({'White NH': 'White', 'Hispanic': 'Hisp.', 'Asian NH': 'Asian', 'Black NH': 'Black',
                             'DK/Refused': 'Don\'t\nKnow', 'AIAN NH': 'AIAN', 'Other NH': 'Other',
                             'NHOPI NH': 'NHOPI'})

# Simplify into boolean column.
df['unemployed'] = df['employ'].str.contains('Unemployed') | df['employ1'].str.contains('unemployed')

# Rename value for nicer plotting.
df['income2'].replace('Don\'t Know', 'Don\'t\nKnow', inplace=True)

# King County's median income in 2015 was $75k
df['over75'] = df['income2'] == '$75+'

# Simplify to boolean.
bmi_dict = {'Over': True, 'Obese': True, 'Normal': False, 'Under': False}
df['overweight'] = df['_bmi5cat'].map(bmi_dict)

# Clean zipcode column.
df['zips'] = df['zipcode'] + df['zipcode1']
df['zips'] = df['zips'].str.strip()
df['zips'] = [int(x) if x != '' else np.nan for x in df['zips']]


def seattle_zip(x):
    if not re.match(r'98\d\d\d[.]0', str(x)):
        return 99999
    else:
        return x


# Remove any entries with zipcodes outside of the Seattle area.
df['zips'] = df['zips'].map(seattle_zip)
df = df[(df['zips'] != 99999) & (df['zips'] != 77777)]

# USDA food desert locations are at the census tract level and can be found at:
# https://www.ers.usda.gov/data-products/food-access-research-atlas
#
# Zipcodes cross referenced with census tracts designated as food deserts can be found at:
# https://www.huduser.gov/portal/datasets/usps_crosswalk.html

zips = [98103, 98115, 98166, 98146, 98188, 98138, 98168, 98178, 98057, 98055, 98058, 98198, 98002, 98001, 98047, 98047]

desert_zips = list(set(zips))

df['desert'] = df['zips'].map(lambda x: True if x in desert_zips else False)

# Simplify to boolean.
health_dict = {'Very Good': True, 'Good': True, 'Excellent': True, 'Fair': False, 'Poor': False, 'DK': False, 'Missing': True}
df['good-health'] = df['genhlth'].map(health_dict)

# Simplify to boolean.
df['prediab1'] = df['prediab1'].fillna('DK')
df['pre-diab'] = df['prediab1'].map(lambda x: True if x == 'Yes' else False)

df['diabetic'] = df['diabete3'].map(lambda x: True if x in ['Yes', 'Borderline/Pre-Diabetes'] else False)

# Simplify to boolean.
df['_paindex'] = df['_paindex'].map(lambda x: True if x == 'Meets Aerobic Recs' else False)
df['_paindx1'] = df['_paindx1'].map(lambda x: True if x == 'Meets Aerobic Recs' else False)

df['active'] = ((df['_paindex'] is True) | df['_paindx1'] is True)

# Clean vegetable and fruit data to reflex frequency of consumption by day, week and month.

# Any unknowns we assume are zero to avoid inflating our good eating habits.
df['vegeda1_'] = df['vegeda1_'].fillna(0)

df['vegeda2_'] = df['vegeda2_'].fillna(0)

df['frutda1_'] = df['frutda1_'].fillna(0)

df['frutda2_'] = df['frutda2_'].fillna(0)

df['ftjuda1_'] = df['ftjuda1_'].fillna(0)

df['ftjuda2_'] = df['ftjuda2_'].fillna(0)

df['_vegesum'] = df['_vegesum'].fillna(0)

df['_vegesu1'] = df['_vegesu1'].fillna(0)

df['_frutsum'] = df['_frutsum'].fillna(0)

df['_frutsu1'] = df['_frutsu1'].fillna(0)

# Combine columns from different years.
df['veg_day'] = df['vegeda1_'] + df['vegeda2_']

df['fruit_day'] = df['frutda1_'] + df['frutda2_']

df['juice_day'] = df['ftjuda1_'] + df['ftjuda2_']

df['veg_sum'] = df['_vegesum'] + df['_vegesu1']

df['fruit_sum'] = df['_frutsum'] + df['_frutsu1']

# Assume survey takers skipping the fruit and veg every day questions are filled with shame at not doing so.
df['_frtlt1'] = df['_frtlt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df['_frtlt1a'] = df['_frtlt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df['_veglt1'] = df['_veglt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df['_veglt1a'] = df['_veglt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

# Combine years.
df['fruit_everyday'] = (df['_frtlt1'] == 1) | (df['_frtlt1a'] == 1)

df['veg_everyday'] = (df['_veglt1'] == 1) | (df['_veglt1a'] == 1)

# Exclude people claiming to eat over 16 servings of fruit and 23 servings of vegetables.
df = df[((df['_vegetex'] == 'Included') | (df['_vegete1'] == 'Included')) &
        ((df['_vegetex'] == 'Included') | (df['_vegete1'] == 'Included'))]


# Functions to decode weekly and monthly frequency responses.
def nothing(x):
    # code 555
    if re.match(r'^555$', str(x)):
        return True
    else:
        return False


def per_week(x):
    # Code in the 200s.
    if re.match(r'^2\d\d[.]0$', str(x)):
        return True
    else:
        return False


def per_week_17(x):
    # Code in the 200s, but not 200.
    if re.match(r'^2\d[1-9][.]0$', str(x)):
        return True
    else:
        return False


def per_month(x):
    # Code in the 300s.
    if re.match(r'^3\d\d[.]0$', str(x)):
        return True
    else:
        return False


def per_month_17(x):
    # Code in the 300s, but not 300.
    if re.match(r'^3\d[1-9][.]0$', str(x)):
        return True
    else:
        return False


# Make a column for no fruit or veg at all.
df['no_veg'] = df['vegetab1'].map(nothing) | df['vegetab2'].map(nothing)
df['no_fruit'] = df['fruit1'].map(nothing) | df['fruit2'].map(nothing)

# And one for some each week, but not everyday. No double counting.
df['veg_week'] = df['vegetab1'].map(per_week) | df['vegetab2'].map(per_week_17)
df['veg_every_week'] = (df['veg_week'] == 1) & (df['veg_everyday'] == 0)

df['fruit_week'] = df['fruit1'].map(per_week) | df['fruit2'].map(per_week_17)
df['fruit_every_week'] = (df['fruit_week'] == 1) & (df['fruit_everyday'] == 0)

# And one for monthly intake, unless daily or weekly boxes are checked. No double counting.
df['veg_month'] = df['vegetab1'].map(per_month) | df['vegetab2'].map(per_month_17)
df['veg_every_month'] = ((df['veg_month'] == 1) & (df['veg_week'] == 0) & (df['veg_everyday'] == 0))

df['fruit_month'] = df['fruit1'].map(per_month) | df['fruit2'].map(per_month_17)
df['fruit_every_month'] = ((df['fruit_month'] == 1) & (df['fruit_everyday'] == 0) & (df['fruit_week'] == 0))

# The columns we want to keep in some sort of order.
final_colunns = ['year', '_ageg5yr', '_race', 'income2', 'over75', 'adults', 'children', 'total_household', 'zips',
                 'desert', 'unemployed', 'active', 'overweight', 'pre-diab', 'diabetic', 'good-health', 'no_veg',
                 'no_fruit', 'fruit_everyday', 'veg_everyday', 'fruit_every_week', 'veg_every_week',
                 'fruit_every_month', 'veg_every_month']

df = df[final_colunns]

# Nicer names for plotting.
final_names = ['Year', 'Age', 'Race', 'Income', 'Over Median Income', 'Adults', 'Children', 'Household Size',
               'Zip-code', 'In Food Desert', 'Unemployed', 'Active', 'Overweight', 'Pre-Diabetic', 'Diabetic',
               'Good-health', 'No Veg', 'No Fruit', 'Fruit Daily', 'Veg Daily', 'Fruit Weekly', 'Veg Weekly',
               'Fruit Monthly', 'Veg Monthly']

df.columns = final_names

# Finally, separate into different years.
df11 = df[df['Year'] == 2011]

df13 = df[df['Year'] == 2013]

df15 = df[df['Year'] == 2015]

df17 = df[df['Year'] == 2017]
