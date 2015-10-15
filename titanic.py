import csv as csv
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy import *

useFeatures = ['Pclass','Sex','Fare','Age']
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

from sklearn.cross_validation import StratifiedShuffleSplit
kf = KFold(len(df), n_folds=100)
validator = StratifiedShuffleSplit(lab, 20, test_size=0.5, random_state=42)
scoreData = []

for train_index, test_index in validator:
    ftr = array(df[useFeatures])
    lab = array(df['Survived'])
    clf = RandomForestClassifier(criterion='gini')
    clf.fit(ftr[train_index], lab[train_index])
    scoreData.append(clf.score(ftr[test_index], lab[test_index]))

print('Mean Training Score: ' + str(np.mean(scoreData)))

#dfTest = pd.DataFrame.from_csv('test.csv')
#dfTest = prepData(dfTest)
#dfTest['Survived'] = clf.predict(array(dfTest[useFeatures]))
#dfTest['Survived'].to_csv('pred.rfc.ClassGenderFare.csv')





