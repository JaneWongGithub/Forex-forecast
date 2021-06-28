

import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
import yfinance as yf
import datetime
from yahoofinancials import YahooFinancials

st.title('📈 Automated FOREX USD-AUD Forecasting')

"""
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 


"""
### Step 1: Upload Live Data directly from Yahoo Financials
"""
import pandas_datareader as pdr
from datetime import date
current_date = date.today()
import matplotlib.pyplot as plt

#define variable for start and end time

# data obtained from Yahoo Financials
#define variable for start and end time
start = datetime(2007, 1, 1)
end = current_date
USDAUD_data = yf.download('AUD=X', start, end)
plt.figure(figsize=(10, 7))
plt.plot(USDAUD_data)       
plt.title('USDAUD Prices')
                                

"""
### Step 2: Select Forecast Horizon

Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 365)

#model for fbprophet:
m = Prophet(daily_seasonality=True, weekly_seasonality=True,yearly_seasonality=True)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_country_holidays(country_name='AU')
m.fit(USDAUD2)

#predicting for the next 7 days from 2019-04 to 2019-08
future = m.make_future_dataframe(periods=7, include_history=True)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""

    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)

model = NeuralProphet(n_changepoints=100,
                      trend_reg=0.05,
                      yearly_seasonality=False,
                      weekly_seasonality=False,
                      daily_seasonality=False)

metrics = model.fit(USDAUD2, validate_each_epoch=True, 
                    valid_p=0.2, freq='D', 
                    plot_live_loss=True, 
                    epochs=100)

"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
