# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:57:37 2017

@author: agupta8
"""
import pandas as pd
import numpy as np
from pandas import read_csv
#from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas.tools.plotting import autocorrelation_plot

#from sklearn.metrics import mean_squared_error

#for Arima
from statsmodels.tsa.arima_model import ARIMA

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

#read the csv file - Acura MDX Sample for Cars, Light Vans Engines 
dataSeries = read_csv('AcuraMDX_Airbags_Sample.csv', parse_dates=['RECALL_DATE'], index_col='RECALL_DATE',date_parser=dateparse)

#visualize the dataset
print(dataSeries.head)
dataSeries.plot()
pyplot.show()

#convert to float to avoid type casting error.
ts = dataSeries['UNIT_AFFECTED_NBR'].astype(np.float64) 

#check the auto-correlation plot for the time series. - Result : high negative correlation untill 1.5
autocorrelation_plot(dataSeries)
pyplot.show()

# fit the model now

model = ARIMA(ts, order = (2,0,0)) #high correlations until level 2

model_fit = model.fit(disp = 0)

print(model_fit.summary())

# plot the residual errors.
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

# density plot of the residual error values, suggesting the errors are Gaussian

residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


#Prediction Set ########################################
Predicted_data_Series = ts.values
Training_Size = int(len(Predicted_data_Series) * .90)
train = Predicted_data_Series[0:Training_Size]
test = Predicted_data_Series[Training_Size:len(Predicted_data_Series)]

Actuals = [Predicted_data_Series for Predicted_data_Series in train]

predictions = list()

for i in range(len(test)):
    output = model_fit.forecast()
    #now import the prediction Yhat inside the empty list
    yhat = output[0]
    predictions.append(yhat)
    obs = test[i]
    Actuals.append(obs)
    print('predicted=%f, Actuals=%f' % (yhat, obs))
#error = mean_squared_error(test, predictions)
#print('Test MSE: %.3f' % error)
# plot
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()




