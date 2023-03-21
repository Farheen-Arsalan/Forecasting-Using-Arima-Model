# time series analysis 
from statsmodels.tsa import seasonal 
import statsmodels.api as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pmdarima import auto_arima
# load the dataset 
path = '/Users/Arsalan/Desktop/AirPassengers.csv'
data = pd.read_csv(path)
print(data.shape)
print(data)

#data cleaning 
data.dropna(axis=0,inplace=True)
# convert the Month into datetime 
data['Month'] = pd.to_datetime(data['Month'])
print(data)
# now set your Month as index 
data.set_index('Month',inplace=True)
print(data)

plt.figure(1)
data.plot(kind='line')

report = seasonal.seasonal_decompose(data)
plt.figure(2)
report.plot()

trData = data['Passengers'] # passanger column
originalData = trData.copy()

#model1 = auto_arima(trData,trace=True,seasonal=True, random_state=10,m=12)

num_sample = 24
for i in range(num_sample):
    # seasonal  SARIMA model for forecasting 
    sarimaModel = st.tsa.statespace.SARIMAX(trData,order=(2,1,1),seasonal_order=(0,1,0,12))
    
    # trained the sarima model
    sarimaModel = sarimaModel.fit()    
    # forcasted output 
    op = sarimaModel.forecast()
    trData=pd.concat([trData,op]) # concatenated the forecasted value in the
    # data itself

print(trData)
plt.figure(100)
plt.plot(trData,c='y')
plt.plot(originalData,c='k')


sarimaModel.plot_diagnostics(figsize=(15,12))










