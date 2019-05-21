import pandas as pd
import numpy as np
import seaborn as sns
import re

pd.set_option('display.max_columns', 500)


columns = ['YEAR', 'NUMADULT', 'HHADULT', 'SEX', '_AGEG5YR', '_PRACE1', 'CHILDREN', '_CHLDCNT', 'EMPLOY', 'EMPLOY1',
           'INCOME2', '_BMI5CAT', 'GENHLTH', 'PREDIAB1', 'DIABETE3', '_PAINDEX', '_PAINDX1', 'TSDROVE']

drop_from_both = [column.lower() for column in ['NUMADULT', 'HHADULT', '_CHLDCNT', 'EMPLOY', 'EMPLOY1', '_BMI5CAT', 'GENHLTH',
                  'PREDIAB1', 'DIABETE3', '_PAINDEX', '_PAINDX1']]

columns_2017 = ['FRUITJU2', 'FRUIT2', 'FVGREEN1', 'VEGETAB2', 'FRENCHF1', 'POTATOE1', 'FTJUDA2_', 'FRUTDA2_',
                'GRENDA1_', 'VEGEDA2_', 'FRNCHDA_', 'POTADA1_', '_MISFRT1', '_MISVEG1', '_FRTRES1',  '_VEGRES1',
                '_FRUTSU1', '_VEGESU1', '_FRTLT1a', '_VEGLT1a', '_VEGETE1', 'ZIPCODE1', '_FRUITE1']

drop_from_2017 = ['FRUITJU1', 'FRUIT1','FVGREEN', 'VEGETAB1', 'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_',
                  'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP','_FRUTSUM',
                  '_VEGESUM', '_FRTLT1', '_VEGLT1', '_FRUITEX', '_VEGETEX', 'ZIPCODE']

drop_from_2017 = [column.lower() for column in drop_from_2017]
columns = [column.lower() for column in columns]
columns_2017 = [column.lower() for column in columns_2017]
all_columns = columns + columns_2017 + drop_from_2017


df_all_years = pd.read_stata("C:\\Users\Paul\PycharmProjects\BlogPost\data\WA_BRFSS_11to17_B.dta",
                             columns=all_columns, convert_missing=False)

df_all_years['numadult'] = df_all_years['numadult'].map(lambda x: 1 if x in [77, 99] else x)
df_all_years['hhadult'] = df_all_years['hhadult'].map(lambda x: 1 if x in [77, 99] else x)

df_all_years['numadult'] = df_all_years['numadult'].fillna(1)
df_all_years['hhadult'] = df_all_years['hhadult'].fillna(1)

df_all_years['adults'] = np.maximum(df_all_years['hhadult'], df_all_years['numadult'])
df_all_years['adults'] = df_all_years['adults'].map(lambda x: 1 if x == 0 else x)

df_all_years['sex'] = df_all_years['sex'].map({9: 2, 'Male': 0, 'Female': 1})

df_all_years['children'] = df_all_years['children'].map(lambda x: 0 if x in [88, 99] else x)

df_all_years['total_household'] = df_all_years['adults'] + df_all_years['children']

df_all_years['unemployed'] = df_all_years['employ'].str.contains('Unemployed') | df_all_years['employ1'].str.contains('unemployed')

bmi_dict = {'Over': True, 'Obese': True, 'Normal': False, 'Under': False}
df_all_years['overweight'] = df_all_years['_bmi5cat'].map(bmi_dict)

df_all_years['zips'] = df_all_years['zipcode'] + df_all_years['zipcode1']
df_all_years['zips'] = df_all_years['zips'].str.strip()
df_all_years['zips'] = [int(x) if x != '' else np.nan for x in df_all_years['zips']]

health_dict = {'Very Good': True, 'Good': True, 'Excellent': True, 'Fair': False, 'Poor': False, 'DK': False, 'Missing': True}
df_all_years['good-health'] = df_all_years['genhlth'].map(health_dict)

df_all_years['pre-diab'] = df_all_years['prediab1'].map(lambda x: True if x == 'Yes' else False)

df_all_years['diabetic'] = df_all_years['diabete3'].map(lambda x: True if x in ['Yes', 'Borderline/Pre-Diabetes'] else False)

df_all_years['_paindex'] = df_all_years['_paindex'].map(lambda x: True if x == 'Meets Aerobic Recs' else False)
df_all_years['_paindx1'] = df_all_years['_paindx1'].map(lambda x: True if x == 'Meets Aerobic Recs' else False)

df_all_years['active'] = ((df_all_years['_paindex'] is True) | df_all_years['_paindx1'] is True)

df_all_years['vegeda1_'] = df_all_years['vegeda1_'].fillna(0)

df_all_years['vegeda2_'] = df_all_years['vegeda2_'].fillna(0)

df_all_years['frutda1_'] = df_all_years['frutda1_'].fillna(0)

df_all_years['frutda2_'] = df_all_years['frutda2_'].fillna(0)

df_all_years['ftjuda1_'] = df_all_years['ftjuda1_'].fillna(0)

df_all_years['ftjuda2_'] = df_all_years['ftjuda2_'].fillna(0)

df_all_years['_vegesum'] = df_all_years['_vegesum'].fillna(0)

df_all_years['_vegesu1'] = df_all_years['_vegesu1'].fillna(0)

df_all_years['_frutsum'] = df_all_years['_frutsum'].fillna(0)

df_all_years['_frutsu1'] = df_all_years['_frutsu1'].fillna(0)

df_all_years['veg_day'] = df_all_years['vegeda1_'] + df_all_years['vegeda2_']

df_all_years['fruit_day'] = df_all_years['frutda1_'] + df_all_years['frutda2_']

df_all_years['juice_day'] = df_all_years['ftjuda1_'] + df_all_years['ftjuda2_']

df_all_years['veg_sum'] = df_all_years['_vegesum'] + df_all_years['_vegesu1']

df_all_years['fruit_sum'] = df_all_years['_frutsum'] + df_all_years['_frutsu1']

df_all_years['_frtlt1'] = df_all_years['_frtlt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df_all_years['_frtlt1a'] = df_all_years['_frtlt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df_all_years['_veglt1'] = df_all_years['_veglt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

df_all_years['_veglt1a'] = df_all_years['_veglt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})
# TODO:  fix the everyday from above
df_all_years['fruit_everyday'] = (df_all_years['fruit_day'] > 0) & (df_all_years['fruit_sum'] < 16)

df_all_years['veg_everyday'] = (df_all_years['veg_day'] > 0) & (df_all_years['veg_sum'] < 23)


def per_week(x):
    if re.match(r'^2\d\d[.]0$', str(x)):
        return True
    else:
        return False


def per_week_17(x):
    if re.match(r'^2\d[1-9][.]0$', str(x)):
        return True
    else:
        return False


def per_month(x):
    if re.match(r'^3\d\d[.]0$', str(x)):
        return True
    else:
        return False


def per_month_17(x):
    if re.match(r'^3\d[1-9][.]0$', str(x)):
        return True
    else:
        return False


df_all_years['veg_every_week'] = (df_all_years['vegetab1'].map(per_week) | df_all_years['vegetab2'].map(per_week_17))

df_all_years['fruit_every_week'] = (df_all_years['fruit1'].map(per_week) | df_all_years['fruit2'].map(per_week_17))

df_all_years['veg_every_month'] = (df_all_years['vegetab1'].map(per_month) | df_all_years['vegetab2'].map(per_month_17))

df_all_years['fruit_every_month'] = (df_all_years['fruit1'].map(per_month) | df_all_years['fruit2'].map(per_month_17))

df11 = df_all_years[df_all_years['year'] == 2011]

df13 = df_all_years[df_all_years['year'] == 2013]

df15 = df_all_years[df_all_years['year'] == 2015]

df17 = df_all_years[df_all_years['year'] == 2017]

df11 = df11.drop(columns_2017 + drop_from_both + drop_from_2017, axis=1)
df13 = df13.drop(columns_2017 + drop_from_both + drop_from_2017, axis=1)
df15 = df15.drop(columns_2017 + drop_from_both + drop_from_2017, axis=1)
df17 = df17.drop(drop_from_2017 + drop_from_both + columns_2017, axis=1)

dataframes = [df11, df13, df15, df17]
