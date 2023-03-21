# time series analysis 
from statsmodels.tsa import seasonal 
import statsmodels.api as st

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# load the dataset 
path = '/Users/Arsalan/Desktop/monthly-milk-production-pounds-p.csv'
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

trData = data['Monthly milk production'] # passanger column
originalData = trData.copy()

num_sample = 12
for i in range(num_sample):
    # seasonal  SARIMA model for forecasting 
    sarimaModel = st.tsa.statespace.SARIMAX(trData,order=(4,2,2),seasonal_order=(1,1,1,12))
    
    # trained the sarima model
    sarimaModel = sarimaModel.fit()    
    # forcasted output 
    op = sarimaModel.forecast()
    trData=pd.concat([trData,op]) # concatenated the forecasted value in the
    # data itself

print(trData)


plt.plot(trData,c='y')
plt.plot(originalData,c='k')

