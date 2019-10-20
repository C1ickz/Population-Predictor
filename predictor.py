import pandas as pd
import matplotlib.pyplot as plt

# Housing data taken from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
df = pd.read_csv('US_births_2000-2014_SSA.csv')
# print(df[['year', 'month', 'date_of_month']].head(10))
convertToDates = pd.DataFrame({'year': df['year'],
                               'month': df['month'],
                               'day': df['date_of_month']})
dates = pd.to_datetime(convertToDates)  # Converts dates to datetime format

df1 = pd.DataFrame({'dates': dates, 'births': df['births']})  # Create new dataframe
summedMonths = df1.set_index('dates').groupby(pd.Grouper(freq='M'))[
    'births'].sum().reset_index()  # Sum each months births

plt.style.use('dark_background')
summedMonths["births"].plot()
plt.title("Births in US from 2000-2014")
plt.ylabel("Births")
plt.show()
# TODO add more descriptive variable names and organize data
