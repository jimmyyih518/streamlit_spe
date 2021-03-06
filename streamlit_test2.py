# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:14:55 2021

@author: JZ2018
"""

import streamlit as st
import pandas as pd
import altair as alt
import datetime
import urllib
import plotly.express as px
from gsheet_fun import *
from electricity_scrape import *
#from pycaret.regression import *

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

#https://docs.google.com/spreadsheets/d/1nf6qwqHwficHPX4gB0mMJtEm3YE31VJHgJtZ8teJJGI/edit?usp=sharing
#@st.cache
#d2 = pd.read_csv('aeso_hpp.csv')
#d2 = d2.drop(d2.columns[0], axis = 1)
#Gsheet_Append(d2, aeso_hpp_id, sheet_range)
#aeso_hpp = Gsheet_Download(aeso_hpp_id, sheet_range)
#aeso_hpp['Date (HE)'] = aeso_hpp['Date (HE)'].apply(pd.to_datetime)
#dftest = aeso_hpp[0:2]
#Gsheet_Append(dftest, aeso_hpp_id, sheet_range)
#test_model = load_model('final_model1')

#function to retrieve data from google sheet
def get_data(sheet_id, sheet_range):
    #df = pd.read_csv('aeso_hpp.csv')
    # here enter the id of your google sheet
    #aeso_hpp_id = '1sRkTyY8jlv-NGizn-0ulBSIgIjQmVvpBVnofy49-NPM'
    #weather_daily_id = '1niPYt8HCYKWLFqnJbSv-7p-kuk9qheIx-igeL5hIy0s'
    #weather_hourly_id = '1k_41_j8CpYeGDNdRWRlIyx_2cx58ROp-6bQ3RcFZidg'
    #sheet_range = 'A1:AA1000000'
    
    #download google sheet with sheet id and excel style range (everything in string format)
    df = Gsheet_Download(aeso_hpp_id, sheet_range)
    #convert datetime column to appropriate format
    df['DATETIME'] = df['DATETIME'].apply(pd.to_datetime)
    #convert numerical columns to appropriate format
    #num_cols = ['Price ($)', '30Ravg ($)', 'AIL Demand (MW)']
    num_cols = ['PRICE']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors = 'coerce')
    # if df.columns.values[0] == 'Unnamed: 0':
    #     df.columns.values[0] = 'RowID'
    # else:
    #     df['RowID'] = df.index+1
    return df

def uca_price_conversion(price):
    consumer_price = 1.1104*price/1000 + 0.0116
    return consumer_price

#function to append new data to google sheet
def append_newdata(data, sheetid, sheet_range):
    #get max date available in downloaded google sheet data
    maxdate = max(data['DATETIME'])
    #get current date time
    currdate = datetime.datetime.now()
    dateformat = '%Y-%m-%d'
    
    #use existing scrape functions to get new data if current date < maxdate in google sheet
    if maxdate < currdate - datetime.timedelta(days=2):
        #call scrape function from electricity_scrape
        append_data = aeso_download_range('HistoricalPoolPrice', 'html',  maxdate.strftime(dateformat),  currdate.strftime(dateformat), dateformat)
        #convert column types
        append_data['Date (HE)'] = append_data['Date (HE)'].apply(pd.to_datetime)
        num_cols = ['Price ($)', '30Ravg ($)', 'AIL Demand (MW)']
        append_data[num_cols] = append_data[num_cols].apply(pd.to_numeric, errors='coerce')
        #append together new scraped data with historical data from google sheet
        append_data2 = append_data[['Date (HE)', 'Price ($)']]
        append_data2.columns = ['DATETIME','PRICE']
        data2 = data.append(append_data2).reset_index(drop=True).drop_duplicates(subset='DATETIME', keep='last').sort_values('DATETIME')
        
        #convert everything to string to prepare for google sheet upload
        upload_data = data2.copy()
        upload_data['Date'] = upload_data['DATETIME'].dt.date
        upload_data_sum = upload_data.groupby('Date').agg({'PRICE':'mean'}).reset_index()
        upload_data_sum.columns = ['DATETIME','PRICE']
        #Gsheet_Append(upload_data, sheetid, sheet_range)
        #upload new data to replace all data in google sheet
        Gsheet_updateAll(upload_data_sum.applymap(str), sheetid, sheet_range)
        print(str(upload_data_sum.shape[0]) + ' rows added to google sheet data')
        
    else:
        data2 = data.copy()
    return data2


# def Gen_Pred_df(model, data):
#     currdate = datetime.datetime.now()
#     maxdate = max(data['Date (HE)'])
df_preds = pd.read_parquet('combine_preds2.parquet')    
df_preds['Date'] = df_preds['Date'].apply(pd.to_datetime)
try:
    #google sheet id for electricity prices and demand (from Historical Pool Price)
    aeso_hpp_id = '1nf6qwqHwficHPX4gB0mMJtEm3YE31VJHgJtZ8teJJGI'
    #aeso_hpp_id = '1sRkTyY8jlv-NGizn-0ulBSIgIjQmVvpBVnofy49-NPM'
    #define max sheet ranges to look for data in google sheet
    sheet_range = 'A1:AA1000000'
    #get data from google sheet
    data_gsheet = get_data(aeso_hpp_id, sheet_range)
    #check if new data needs to be appended to google sheet
    data_new = append_newdata(data_gsheet, aeso_hpp_id, sheet_range)
    #clean and sort data
    data_new2 = data_new.reset_index(drop=True).drop_duplicates().sort_values('DATETIME')
    data_new2.columns = ['Date', 'Predicted Price $']
    data_new2['Data Source'] = 'AESO_HPP_Historical'
    data = df_preds.append(data_new2).sort_values(['Data Source', 'Date']).reset_index(drop=True)
    data_models = data['Data Source'].unique().tolist()
    data_models.remove('AESO_HPP_Historical')
    #data['Date'] = data['Date'].apply(pd.to_datetime)
    data['Electricity Price $/kwh'] = uca_price_conversion(data['Predicted Price $'])
    #data = data.rename(columns={'Date (HE)':'Date'})
    #data['Date'] = data['Date'].apply(pd.to_datetime)
    
    #print out current datetime in streamlit
    st.write('Current Date '+str(datetime.datetime.now().strftime('%Y-%m-%d')))    
    #streamlit input to filter plot range by minimum date
    mindate = st.date_input(
       'Plot Start Date', datetime.date(2020,1,1)
    )
    #streamlit input to update plot with defined range
    # st.write('Click below to update plot time range')
    # dateupdate = st.button('Update Plot Time Range')
    # #streamlit input for user's input electricity price quote
    st.sidebar.write('Enter your quoted electricity price below in $/kwh')
    user_priceinput = st.sidebar.number_input('Price', value = 0.05)
    st.sidebar.write(f'Your price is {user_priceinput} $/kwh')
    #streamlit input for user's input electricity price quote period
    st.sidebar.write('Enter your electricity price lock-in time')
    #user_locktime = st.number_input('Years', format = '%i')
    user_locktime = st.sidebar.slider('Years', 0, 5, 1)
    user_locktime_int = int(user_locktime)
    st.sidebar.write(f'Your quoted price is locked in for {user_locktime_int} years')
    
    user_select_mod = st.sidebar.selectbox(label = 'Select Model', options=data_models)
    
    #convert minimum plot range date to appropriate format
    mindate = datetime.datetime.strptime(str(mindate), '%Y-%m-%d')
    maxdate = datetime.datetime.now() + datetime.timedelta(days=user_locktime_int*365)
    maxdate = maxdate.strftime('%Y-%m-%d')
    maxdate = datetime.datetime.strptime(str(maxdate), '%Y-%m-%d')
    #execute only if "mindate" exists
    if not mindate:
        st.error("Please select start date for plot range")
    else:
        # #update plot ranges via filtering the table to dates newer than mindate only
        # if dateupdate:
        #     df = data.loc[data['Date']>=mindate]
        # else:
        #     df = data.loc[data['Date']>=datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')]
        df = data.loc[data['Date']>=mindate]
        df = df.loc[df['Date']<=maxdate]
        df['DailyCost'] = df['Electricity Price $/kwh'] * 20
        df_calcs = df[df['Data Source'] == user_select_mod]
        df_calcs = df_calcs.loc[df_calcs['Date']>=datetime.datetime.now().strftime('%Y-%m-%d')]
        df_calcs['CumCost'] = df_calcs['DailyCost'].cumsum()
        user_cost = 7300*user_priceinput*user_locktime_int
        cost_diff = round(df_calcs['CumCost'].max() - user_cost, 2)
        if cost_diff > 0:
            out_txt = f'Your Quote is predicted to save ${abs(cost_diff)} dollars compared to variable price over {user_locktime_int} years'
            st.markdown(f'<font color="green">{out_txt}</font>', unsafe_allow_html=True)
        else:
            out_txt = f'Your Quote is predicted to cost ${abs(cost_diff)} more dollars compared to variable price over {user_locktime_int} years'
            st.markdown(f'<font color="red">{out_txt}</font>', unsafe_allow_html=True)
        
        #write a header line and dataframe for visualization
        # st.write("AESO Historical Data", df)

        # #test Altair charting with electricity price over time
        # chart1 = (
        #     alt.Chart(df)
        #     .mark_line(opacity=0.5)
        #     .encode(
        #         x="Date:T",
        #         y=alt.Y("Electricity Price $/kwh:Q", stack=None)
        #     )
        # )
        
        # chart2 = (
        #     alt.Chart(df)
        #     .mark_line(opacity=0.5)
        #     .encode(
        #         x="Date:T",
        #         y=alt.Y("AIL Demand (MW):Q", stack=None)
        #     )
        # )
        
        #test plotly charting with AIL Demand over time
        # chart2 = px.scatter(df, x='Date', y='AIL Demand (MW)')
        
        # st.write('Alberta Electricity Price History')
        # st.altair_chart(chart1, use_container_width=True)
        # st.write('Alberta Electricity Demand Hsitory')
        # st.plotly_chart(chart2, use_container_width=True)
        
        fig=go.Figure()
        plt_types = df['Data Source'].unique().tolist()
        for p in plt_types:
            subdf = df.loc[df['Data Source']==p]
            if p == 'AESO_HPP_Historical':
                plt_mod = 'markers'
            else:
                plt_mod = 'lines'
            fig.add_trace(go.Scatter(x=subdf['Date'], y=subdf['Electricity Price $/kwh'], mode = plt_mod, name = p))
            fig.add_hline(y=user_priceinput)
            fig.update_yaxes(type='log')
            fig.update_layout(width = 900,
                              title = 'Prediction and Historical Prices',
                              xaxis_title = 'Date',
                              yaxis_title = 'Electricity Price $/kwh',
                              legend_title = 'Plot Data Sources',
                              font = dict(family='Courier New, monospace',size=15,color='RebeccaPurple'))
        st.write('Alberta Electricity Demand History')
        st.plotly_chart(fig, use_container_width=False)
       
#error handling function        
except urllib.error.URLError as e:
    st.error(
        """
        **error**

        Connection error: %s
    """
        % e.reason
        )