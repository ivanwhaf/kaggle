import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# preprocess train dataset
train_data = pd.read_csv("train.csv")
train_data['Age'].fillna(0, inplace=True)  # set null age=0
# train_data['Cabin'].fillna('unknown', inplace=True)
train_data['Embarked'].fillna('unknown', inplace=True)

le = preprocessing.LabelEncoder()
le = le.fit(train_data['Embarked'].unique())
embarked = le.transform(train_data['Embarked'])
train_data['Embarked'] = embarked

le = le.fit(train_data['Sex'].unique())
embarked = le.transform(train_data['Sex'])
train_data['Sex'] = embarked

# preprocess train dataset
test_data = pd.read_csv("test.csv")
test_data['Age'].fillna(0, inplace=True)  # set null age=0
# test_data['Cabin'].fillna('unknown', inplace=True)
test_data['Fare'].fillna(0, inplace=True)
test_data['Embarked'].fillna('unknown', inplace=True)

le = preprocessing.LabelEncoder()
le = le.fit(test_data['Embarked'].unique())
embarked = le.transform(test_data['Embarked'])
test_data['Embarked'] = embarked

le = le.fit(test_data['Sex'].unique())
embarked = le.transform(test_data['Sex'])
test_data['Sex'] = embarked

X_train = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train_data["Survived"]
print(X_train)
X_test = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
print(X_test)


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y)
print('train done')
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
