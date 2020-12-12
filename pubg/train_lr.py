import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("pubg/train_V2.csv")
print(train_data.describe())
print(train_data.skew())
print(train_data.isnull().any())
train_data['winPlacePerc'].fillna(value=0, inplace=True)
print(train_data['matchType'].value_counts())

# transform
le = preprocessing.LabelEncoder()
le = le.fit(train_data['matchType'].unique())
matchType = le.transform(train_data['matchType'])
train_data['matchType'] = matchType

X_train = train_data[['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'matchType',
                      'maxPlace', 'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
print(X_train.head())
y = train_data['winPlacePerc']


test_data = pd.read_csv("pubg/test_V2.csv")

# label encoder
le = preprocessing.LabelEncoder()
le = le.fit(test_data['matchType'].unique())
matchType = le.transform(test_data['matchType'])
test_data['matchType'] = matchType

print(test_data.isnull().any())
X_test = test_data[['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'matchType',
                    'maxPlace', 'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
# train model
model = LinearRegression()
model.fit(X_train[:3000000], y[:3000000])
print('train done')
predictions = model.predict(X_test)

output = pd.DataFrame(
    {'Id': test_data.Id, 'winPlacePerc': predictions})
output.to_csv('pubg/submission.csv', index=False)
