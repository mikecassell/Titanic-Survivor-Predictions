import csv as csv
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy import *

useFeatures = ['Pclass','Sex','Fare']
def parse_Fares(fare):
    return(np.floor(fare/10) * 10)
    
def prepData(df):
    df['Embarked'] = pd.get_dummies(df['Embarked'])
    df['Sex'] = pd.get_dummies(df['Sex'])
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))
    df['Fare'] = df['Fare'].apply(lambda q: parse_Fares(q))
    df['Age'] = df['Age'].fillna(np.mean(df['Age']))
    df['Age'] = df['Age'].apply(lambda q: np.floor(q))
    df = df.drop(['Name','Cabin','Ticket'], axis=1)
    return(df)
    
def targetFeatureSplit( data ):
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features

df = pd.DataFrame.from_csv('train.csv')
df = prepData(df)

print(df)

ftr = array(df[useFeatures])
lab = array(df['Survived'])
clf = RandomForestClassifier(criterion='gini')
clf.fit(ftr, lab)

print('Training Score:' + str(clf.score(ftr, lab)))

df = pd.DataFrame.from_csv('test.csv', index_col=None)
df = prepData(df)
ftr = array(df[useFeatures])

pred = clf.predict(ftr)
res = zip(df['PassengerId'], pred)

with open("results.3.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(['PassengerId','Survived'])
    writer.writerows(res)