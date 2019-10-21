import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Housing data taken from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('US_births_2000-2014_SSA.csv')
# print(df[['year', 'month', 'date_of_month']].head(10))
convertToDates = pd.DataFrame({'year': df['year'],
                               'month': df['month'],
                               'day': df['date_of_month']})
dates = pd.to_datetime(convertToDates)  # Converts dates to datetime format

df1 = pd.DataFrame({'dates': dates, 'births': df['births']})  # Create new dataframe
summedMonths = df1.set_index('dates').groupby(pd.Grouper(freq='M'))[
    'births'].sum().reset_index()  # Sum each months births
births = summedMonths['births']
births = births.values.reshape(len(births), 1)
scaler = MinMaxScaler(feature_range=(0, 1))
births = scaler.fit_transform(births)

train_size = int(len(births) * .6)
test_size = len(births) - train_size

births_train = births[0: train_size, :]
births_test = births[train_size:len(births), :]
print(len(births_train), len(births))


def create_ts(ds, series):
    X = []
    Y = []
    for i in range(len(ds) - series - 1):
        item = ds[i:(i + series), 0]
        X.append(item)
        Y.append(ds[i + series, 0])
    return np.array(X), np.array(Y)


series = 10  # How many days ahead being predicted
trainX, trainY = create_ts(births_train, series)
testX, testY = create_ts(births_test, series)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(64, input_shape=(series, 1)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.fit(trainX, trainY, epochs=1000, batch_size=32)
trainPredictions = model.predict(trainX)
testPredictions = model.predict(testX)

trainPredictions = scaler.inverse_transform(trainPredictions)
testPredictions = scaler.inverse_transform(testPredictions)

train_plot = np.empty_like(births)
train_plot[:, :] = np.nan
train_plot[series:len(trainPredictions) + series, :] = trainPredictions

test_plot = np.empty_like(births)
test_plot[:, :] = np.nan
test_plot[len(trainPredictions) + (series * 2) + 1:len(births) - 1, :] = testPredictions

plt.plot(scaler.inverse_transform(births))
# plt.plot(train_plot)
plt.plot(test_plot)
plt.show()
# TODO add more descriptive variable names, organize data, make x axis values by year, save neural network trainin models
