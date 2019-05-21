import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)


columns = ['YEAR', 'NUMADULT', 'HHADULT', 'SEX', '_AGEG5YR',
           '_PRACE1', 'CHILDREN', '_CHLDCNT', 'EDUCA', 'EMPLOY', 'EMPLOY1',
           'INCOME2', '_BMI5CAT',
           'GENHLTH', 'PREDIAB1', 'DIABETE3',
           '_PAINDEX', '_PAINDX1', 'TSDROVE']
columns_2017 = ['FRUITJU2', 'FRUIT2', 'FVGREEN1', 'VEGETAB2', 'FRENCHF1', 'POTATOE1', 'FTJUDA2_', 'FRUTDA2_',
                'GRENDA1_', 'VEGEDA2_', 'FRNCHDA_', 'POTADA1_', '_MISFRT1', '_MISVEG1', '_FRTRES1',  '_VEGRES1',
                '_FRUTSU1', '_VEGESU1', '_FRTLT1a', '_VEGLT1a', '_FRT16a', '_VEG23a','_FRUITE1',
                '_VEGETE1', 'ZIPCODE1']
drop_from_2017 = [ 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'VEGETAB1', 'FTJUDA1_', 'FRUTDA1_',
           'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN', '_FRTRESP',
           '_VEGRESP','_FRUTSUM', '_VEGESUM', '_FRTLT1', '_VEGLT1', '_FRT16', '_VEG23',
           '_FRUITEX', '_VEGETEX', 'ZIPCODE']

drop_from_2017 = [column.lower() for column in drop_from_2017]
columns = [column.lower() for column in columns]
columns_2017 = [column.lower() for column in columns_2017]
all_columns = columns + columns_2017 + drop_from_2017


df_all_years = pd.read_stata("C:\\Users\Paul\PycharmProjects\BlogPost\data\WA_BRFSS_11to17_B.dta",
                             columns=all_columns, convert_missing=False)

df_all_years['adults'] = df_all_years['hhadult'].fillna(1)

df_all_years['sex'] = df_all_years['sex'].map({9: np.nan, 'Male': 0, 'Female': 1})

df_all_years['children'] = df_all_years['children'].map(lambda x: 0 if x in [88, 99] else x)

df_all_years['total_household'] = df_all_years['adults'] = df_all_years['children']

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


# df11 = df_all_years[df_all_years['year'] == 2011]
# # df12 = df_all_years[df_all_years['year'] == 2012]
# df13 = df_all_years[df_all_years['year'] == 2013]
# # df14 = df_all_years[df_all_years['year'] == 2014]
# df15 = df_all_years[df_all_years['year'] == 2015]
# # df16 = df_all_years[df_all_years['year'] == 2016]
# df17 = df_all_years[df_all_years['year'] == 2017]
#
# df11 = df11.drop(columns_2017, axis=1)
# df13 = df13.drop(columns_2017, axis=1)
# df15 = df15.drop(columns_2017, axis=1)
# df17 = df17.drop(drop_from_2017, axis=1)
#
# dataframes = [df11, df13, df15, df17]
#
# new_columns = ['year', 'phone', 'in-state']
