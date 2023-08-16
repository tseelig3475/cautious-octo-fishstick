from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

#training data
train_fp = '/Users/tseelig/scripts/kaggle/house_prices/home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_fp)

#test data
test_fp = '/Users/tseelig/scripts/kaggle/house_prices/home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_fp)

features = ['LotArea', 'OverallCond', 'YearBuilt']
X = train_data[features]
y = train_data.SalePrice

#naive model
housing_model = DecisionTreeRegressor(random_state=1)
housing_model.fit(X, y)

#predict prices with naive model 
predictedPrices = housing_model.predict(test_data[['LotArea', 'OverallCond', 'YearBuilt']])


#splitting up the dataset into train and validate sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
