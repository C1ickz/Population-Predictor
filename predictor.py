import pandas as pd
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os.path

# Birth data: https://github.com/fivethirtyeight/data/tree/master/births


df = pd.read_csv('US_births_1994-2014_SSA.csv')  # reads data from birthData csv
# print(df[['year', 'month', 'date_of_month']].head(10))

# Converts the year, month, and date_of_month file into a new dataframe
convertToDates = pd.DataFrame({'year': df['year'],
                               'month': df['month'],
                               'day': df['date_of_month']})
dates = pd.to_datetime(convertToDates)  # Converts dates to datetime format Ex. (yyyy,mm,dd)

df1 = pd.DataFrame({'dates': dates, 'births': df['births']})  # New dataframe with only necessary data

summedMonths = df1.set_index('dates').groupby(pd.Grouper(freq='M'))[
    'births'].sum().reset_index()  # Sum each months births. Does not need to be used, but makes graph look neater. CAUTION: Could suffer from underfitting
births = summedMonths['births']  # Uses data from the births column to fill new variable
births = births.values.reshape(len(births), 1)  # Makes len(birth) arrays with one value in each
scaler = MinMaxScaler(feature_range=(0, 1))  # Scales data between 0, 1 (Normalize
births = scaler.fit_transform(births)  # Uses MinMaxScaler on births data

train_size = int(len(births) * .6)  # Size of training data
test_size = len(births) - train_size  # Size of prediction

births_train = births[0: train_size, :]  # Use data values 0 to train_size
births_test = births[train_size:len(births), :]  # Use data values from train_size to end of the dataset
print(len(births_train), len(births))


def create_ts(ds, series):
    X = []
    Y = []
    for i in range(len(ds) - series - 1):
        item = ds[i:(i + series), 0]
        X.append(item)
        Y.append(ds[i + series, 0])
    return np.array(X), np.array(Y)


series = 10  # How many days back are being used to predict future trends
trainX, trainY = create_ts(births_train, series)
testX, testY = create_ts(births_test, series)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


hidden_nodes = 64  # mess around with this number to see if you can get the model to be more accurate. Note - The more you add the longer it takes, but it gets more complex
model = Sequential()
model.add(LSTM(hidden_nodes, input_shape=(series, 1)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.fit(trainX, trainY, epochs=100, batch_size=32)
model.save('model.h5')
del model

model = load_model('model.h5')
model.fit(trainX, trainY, epochs=10000, batch_size=32)
model.save('model.h5')


def predictionGen():
    trainPredictions = model.predict(trainX)
    testPredictions = model.predict(testX)

    trainPredictions = scaler.inverse_transform(trainPredictions)
    testPredictions = scaler.inverse_transform(testPredictions)

    train_plot = np.empty_like(
        births)  # creates an array with the same shape and type and fills it with arbitrary values
    train_plot[:, :] = np.nan
    train_plot[series:len(trainPredictions) + series, :] = trainPredictions

    test_plot = np.empty_like(births)
    test_plot[:, :] = np.nan  # fill an array of arrays with nan
    test_plot[len(trainPredictions) + (series * 2) + 1:len(births) - 1, :] = testPredictions

    return train_plot, test_plot


def createGraph(train_data, predicted_data):
    plt.title("Births over time")
    plt.xlabel("Date")
    plt.ylabel("Births")
    plt.plot(summedMonths['dates'], summedMonths['births'])  # plot original data
    plt.plot(summedMonths['dates'], train_data)  # plot training data
    plt.plot(summedMonths['dates'], predicted_data)  # plot prediction
    plt.legend(loc='upper left')  # puts legend in upper left corner of graph
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()


train_data, predicted_data = predictionGen()
createGraph(train_data, predicted_data)
# TODO add more descriptive variable names, organize data, make x axis values by year, save neural network training models
