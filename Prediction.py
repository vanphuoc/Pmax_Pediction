# In[0]: IMPORT AND FUNCTIONS
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold   
from statistics import mean
import joblib 
import pickle
from custom_transformer import ColumnSelector
#%% LOAD raw data from CSV
raw_data_2nd_test = pd.read_csv('Data_Test_1.csv')

# %% REVIEW DATA JUST LOADED
print('\n____________ Dataset info ____________')
print(raw_data_2nd_test.info())  

# %% LOAD Prediction MODEL
search = joblib.load('RandomForestRegressor_model.pkl')
#best_model = search.best_estimator_
best_model = search
# %% Load data transfomer model
full_pipeline = joblib.load('full_pipeline2.pkl')
#%% DATA LOAD
#prepare data
test_2nd_set = raw_data_2nd_test
test_2nd_set_labels = test_2nd_set["Pmax"].copy()
test_2nd_set = test_2nd_set.drop(columns = "Pmax")
print(len(test_2nd_set), "Test +", len(test_2nd_set), "test examples")
print(test_2nd_set.head(4)) 
processed_2nd_test_set_val = full_pipeline.transform(test_2nd_set)  
print('\n____________ Processed feature values ____________')
print(processed_2nd_test_set_val[[0],:].toarray())
print(processed_2nd_test_set_val.shape)
#prediction
#Evaluate function
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse  

r2score_2nd, rmse_2nd = r2score_and_rmse(best_model, processed_2nd_test_set_val, test_2nd_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score_2nd)
print("Root Mean Square Error: ", rmse_2nd.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_2nd_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_2nd_test_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(test_2nd_set_labels[0:9]),'\n')
# %%




#%%

# %%
