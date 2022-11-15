import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from category_encoders import OneHotEncoder
import sklearn.model_selection as model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split




data = pd.read_csv("sales_train.csv")

X = data.iloc[:, :-1]
y = data['item_cnt_day']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



class Preprocessor():
    def __init__(self):
        self.preprocessor = self.get_preprocessor()
        
    def get_preprocessor(self):

        numeric_transformer = Pipeline([
                ("scaler", StandardScaler())
            ])
        
        preprocessor = ColumnTransformer([
            ("numeric", numeric_transformer, [])
        
        ],remainder='passthrough')
        

        return preprocessor
        
    def append_features(self,data):
        '''
        Parameters
        ----------
        data : data to be appended, must have 'date' feature

        Returns
        -------
        data : data appended with new features derived from 'date' column

        '''
        data['date'] = pd.to_datetime(data['date'],format = "%d.%m.%Y")

        data['year'] = data['date'].dt.year
        data['day_of_year'] = data['date'].dt.dayofyear

        data['day_of_week'] = data['date'].dt.dayofweek
        data['week'] = data['date'].dt.week
        # there's no item_price in test ?? 
        data = data.drop(['date','item_price'],axis=1)
        
        return data

    def fit_trainsform(self,X_train):
        X_train = self.append_features(X_train)
        self.preprocessor.fit_transform(X_train)
        return X_train
    
    def transform(self,data):
        data = self.append_features(data)
        #Problem: pipeline returns unexplected outpt when fitting
        #data = self.preprocessor.transform(data)
        return data


prep = Preprocessor()

X_train = prep.fit_trainsform(X_train)
X_test = prep.transform(X_test)

###### XGboost modelling

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import cv
import pickle



train = xgb.DMatrix(X_train, y_train)
test = xgb.DMatrix(X_test, y_test)


print("Fitting phase")

params = {
'colsample_bynode': 0.8,
'learning_rate': 1,
'max_depth': 5,
'num_parallel_tree': 100,
'objective': 'reg:squarederror',
'subsample': 0.8,
}

model_xgb = xgb.train( params, dtrain=train, evals=[(train, "train"), (test, "validation")], num_boost_round=100, early_stopping_rounds=20)


# Plot for report
xgb.plot_importance(model_xgb)
plt.figure(figsize = (16, 12))
plt.show()

# Saving trained model
pickle.dump(model_xgb, open('xgboost_1.0.pkl', "wb"))




# Generating submission data
X_real_test = pd.read_csv("test.csv")
X_real_test['date'] = pd.to_datetime("15.11.2015",format = "%d.%m.%Y")
X_real_test['date_block_num'] = 34

IDs = X_real_test['ID']

X_real_test = X_real_test.drop(['ID'],axis=1)

X_real_test = prep.transform(X_real_test)

predictions = model_xgb.predict(xgb.DMatrix(X_real_test))

results = pd.DataFrame[columns=["ID","item_cnt_month"], ]

predictions.to_csv("results")

