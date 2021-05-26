# -*- coding: utf-8 -*-
"""
Created on Thu May 20 23:14:04 2021

@author: JZ2018
"""

import pandas as pd
import datetime
from pycaret.regression import *
import numpy as np
from matplotlib import pyplot

#df_weather = pd.read_csv('D:/JZP/weather_fcst.csv')
df_weather = pd.read_parquet('data/weather_fcst.parquet')

#df_aeso = pd.read_csv('D:/JZP/aeso_hpp_historical.csv')
df_aeso = pd.read_parquet('data/aeso_hpp_historical.parquet')
df_combined = df_weather.merge(df_aeso, how = 'left', left_on='Datetime', right_on='Date (HE)')
df_combined['Datetime'] = df_combined['Datetime'].apply(pd.to_datetime)
df_combined['day_of_week'] = df_combined['Datetime'].dt.dayofweek.apply(pd.to_numeric, errors='coerce')
#df_combined['day_of_week'] = df_combined['day_of_week']+0.01
df_combined['day_of_year'] = df_combined['Datetime'].dt.dayofyear
df_combined['hour_of_day'] = df_combined['Datetime'].dt.hour
df_combined['year'] = df_combined['Datetime'].dt.year
df_combined['Date'] = df_combined['Datetime'].dt.date
df_combined['logprice'] = np.log10(df_combined['Price ($)'])
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

#gas_df = pd.read_csv('data/gas_df.csv')
gas_df = pd.read_parquet('data/gas_df.parquet')
gas_df['Date'] = gas_df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%b-%y'))
gas_df['yearmon'] = gas_df['Date'].dt.year.astype(str) + '-' + gas_df['Date'].dt.month.astype(str)
df_combined['yearmon'] = df_combined['Datetime'].dt.year.astype(str) + '-' + df_combined['Datetime'].dt.month.astype(str)
df_combined = df_combined.merge(gas_df, how='left', left_on='yearmon', right_on='yearmon')

prod_gen = pd.read_csv('data/electricity_generation_primary_fuel_alberta.csv')
prod_gen['TotalEnergyProduction'] = prod_gen[['Hydro','Wind',
       'Biomass_Geothermal', 'Solar', 'Uranium', 'Coal', 'natural_gas', 'oil']].sum(axis=1)
prod_gen['NatGas_ProdPercent'] = prod_gen['natural_gas'] / prod_gen['TotalEnergyProduction']

df_combined = df_combined.merge(prod_gen, how='left', left_on='year', right_on='year')

df_coal = pd.read_csv('data/combined_coalprice.csv')
df_combined = df_combined.merge(df_coal, how='left', left_on='year', right_on='Year')
df_combined['gas_cost_tot'] = df_combined['AECO_USD_per_MMBtu']*df_combined['natural_gas']
df_combined['coal_cost_tot'] = df_combined['Coal Price'] * df_combined['Coal']

label_col = 'logprice'
feature_cols = ['Min Temp (°C)', 'coal_cost_tot',
                'gas_cost_tot', 'CADUSD'
       ]

ts_cols = ['day_of_week', 'day_of_year', 'hour_of_day']
#ts_cols = ['day_of_week', 'hour_of_day']
df_combined[feature_cols] = df_combined[feature_cols].fillna(method='ffill')
testyear = 2019
df_mod = df_combined[df_combined['Datetime'].dt.date<=datetime.datetime.strptime('2021-03-25', '%Y-%m-%d').date()] 
#df_mod = df_mod.groupby('Date_x').mean()
for col in df_mod.columns:
    if df_mod[col].dtypes == 'int64':
        df_mod[col] = df_mod[col].astype(np.float64)

# def outlier_rm(x, rng):
#     x=x.fillna(method='ffill')
#     med_x = np.median(x)
#     ix = [abs(i) for i in x] > med_x * rng
#     ix = ix.tolist()
#     x_out = [x[i] if ix[i] is False else float('nan') for i in range(0,len(x))]
#     x_outpd = pd.Series(x_out)
#     output = x_outpd.interpolate().tolist()
#     return output

# df_mod[label_col] = outlier_rm(df_mod[label_col], rng=2)

df_train = df_mod[df_mod['year']!=testyear]
df_train = df_train[df_train[label_col].notnull()]
df_test = df_mod[df_mod['year']==testyear]

data_f2 = df_train[[label_col]+feature_cols+ts_cols]
data_f2 = data_f2[data_f2[label_col].notnull()]
#data_f2 = data_f2.loc[data_f2[label_col]>0]

#df_train['month'] = df_train['Datetime'].dt.month
df_plot = df_train.groupby('yearmon').mean().reset_index()
def Zscore(x):
    x_list = x.tolist()
    x_filter = [y for y in x_list if x_list != None]
    z = (x - np.mean(x)) / np.std(x)
    return z

df_plot['zlabel'] = Zscore(df_plot[label_col])
df_plot['zlabel'] = df_plot[label_col].apply(lambda x: Zscore(x))
df_plot['zlabel'] = df_plot[label_col].apply(Zscore)

(df_plot['zlabel']).plot()
df_plot['AECO_USD_per_MMBtu'].plot()
(df_plot['Coal Price']*0.1).plot()

import seaborn as sns

plot_cols = ['Hydro', 'Wind', 'Biomass_Geothermal', 'Solar', 'Uranium',
       'Coal', 'natural_gas', 'oil', 'TotalEnergyProduction']
plot_cols=['CADUSD',
       'Henry_Hub_USD_per_MMBtu', 'AECO_USD_per_MMBtu',
       'Station_2_USD_per_MMBtu', 'NBP_USD_per_MMBtu']
plot_cols = ['AIL Demand (MW)','day_of_week', 'day_of_year', 'hour_of_day', 'year'
    ]


sns.pairplot(df_plot, y_vars = label_col, x_vars = df_plot[[label_col]+feature_cols].columns.values)
g=sns.PairGrid(data_f2)
g.map(sns.scatterplot)



clf1 = setup(data=data_f2, target=label_col, html=False)
cm = compare_models(n_select = 5)
best_model = cm[0]

plot_model(best_model, plot='feature', save=True)
plot_model(best_model, plot='residuals', save=True)


from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(cm[0], X=data_f2[feature_cols+ts_cols], features=feature_cols+ts_cols) 


preds = predict_model(cm[0], data = df_combined)
preds.to_csv('data/newpreds.csv')
save_model(cm[0], 'extra_trees_model')
#best_model = load_model('extra_trees_model')
#preds = pd.read_csv('data/newpreds.csv')
#preds['Datetime'] = preds['Datetime'].apply(pd.to_datetime)
#
predplot = preds.loc[preds['year']==testyear]
predplot['month'] = predplot['Datetime'].dt.month
#predplot = predplot[predplot['month']>=4]
plot_datecol = 'Datetime'
predplot = predplot.groupby(plot_datecol).agg({label_col:'mean',
                                           'Label':'mean'}).reset_index()

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
fig = go.Figure()
fig.add_trace(go.Scatter(x=predplot[plot_datecol], y=predplot['Label'], mode = 'lines', name = 'Predicted'))
fig.add_trace(go.Scatter(x=predplot[plot_datecol], y=predplot[label_col], mode = 'markers', name = 'Actual'))
fig.write_html('test_predicted_actuals.html')

import numpy as np
pred_corr = predplot[[label_col, 'Label']]
pred_corr = pred_corr.dropna()
corr_mtx = np.corrcoef(pred_corr[label_col], pred_corr['Label'])
corr_xy = corr_mtx[0,1]
r_sq = corr_xy**2
print(r_sq)



import matplotlib.pyplot as plt
p1 = plt.figure()
ax = plt.gca()
ax.scatter(pred_corr[label_col], pred_corr['Label'], )
ax.set_xlabel('Actuals')
ax.set_ylabel('Predicted')

#ax.set_yscale('log')
#ax.set_xscale('log')


streamlit_preds = pd.read_parquet('combine_preds2.parquet')
streamlit_preds = streamlit_preds[streamlit_preds['Data Source'] != 'ExtraTrees_NewJZ']
append_preds = preds[['Date_x','Label']].groupby('Date_x').agg({'Label':'mean'}).reset_index()
append_preds.columns = ['Date','Predicted Price $']
append_preds['Data Source'] = 'ExtraTrees_NewJZ'
append_preds['Predicted Price $'] = 10 ** append_preds['Predicted Price $']

streamlit_final_preds = streamlit_preds[streamlit_preds['Data Source'].isin(['AESO_HPP_Historical', 'RandomForestRegressor_JZ', 'FBProphet_JZ'])]
streamlit_final_preds = streamlit_final_preds.append(append_preds)
streamlit_final_preds['Predicted Price $'] = streamlit_final_preds['Predicted Price $'].apply(pd.to_numeric, errors='coerce')
#streamlit_final_preds['Predicted Price $'] = streamlit_final_preds['Predicted Price $'].fillna(method='ffill')
streamlit_final_preds['Date'] = streamlit_final_preds['Date'].apply(pd.to_datetime).dt.date
#streamlit_final_preds['Date'] = streamlit_final_preds['Date'] .apply(pd.to_datetime)
#streamlit_final_preds.to_csv('combine_preds2.csv')
streamlit_final_preds.to_parquet('combine_preds2.parquet')

