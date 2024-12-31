from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
import xgboost as xgb
import pickle

Xtrain = pd.read_excel('Xtrain_new.xlsx')    
Xtest = pd.read_excel('Xtest_new.xlsx')
Ytrain = pd.read_excel('ytrain_lg.xlsx')
Ytest = pd.read_excel('ytest_lg.xlsx')

dtrain = xgb.DMatrix(Xtrain,Ytrain)
dtest = xgb.DMatrix(Xtest,Ytest)

# model parameter seting 
param = {'silent': True
          ,'booster':'gbtree'
          ,"subsample": 0.8  
          ,"max_depth": 7 
          ,"eta": 0.05                            
          ,"alpha": 0.01                           
          ,"colsample_bytree": 0.9
          ,'min_child_weight': 8
          ,"nfold": 5}
num_round = 500

watchlist = [(dtrain,'train'), (dtest,'eval')]
evals_result = {}
bst = xgb.train(param, dtrain, num_round,watchlist, evals_result=evals_result, early_stopping_rounds=100, verbose_eval=False)

# save model structure 
pickle.dump(bst, open("xgb_INMLac20241024.dat","wb")) 

# import model
loaded_model = pickle.load(open("xgb_INMLac20241024.dat", "rb"))
print("Loaded model from: xgb_INML.dat")

# 30% dataset testing 
ypreds = loaded_model.predict(dtest)

y_test=np.power(10,Ytest)
ypreds=np.power(10,ypreds)

r2_score1 = r2_score(y_test,ypreds)
rmse1 = MSE(ypreds, y_test,squared=False)
MAPE1 = np.mean(np.abs(ypreds - y_test) / y_test)*100
bias = np.average(np.log10(y_pred) - np.log10(y_test),axis=0)
bias1 = np.power(10,bias)

print(r2_score1)
print(rmse1)
print(MAPE1)
print(bias1)

# Export figure about 30% testing
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
fig = plt.figure(facecolor='white')
plt.scatter(y_test, ypreds, color=(234.0/255, 11.0/255, 11.0/255),s=35, alpha=0.6
            ,edgecolors='black'
           )
plt.xlabel('Measured Salinity (ppt)')
plt.ylabel('Estimated Salinity (ppt)')
plt.title('XGB model')

plt.xlim(-1, 25)
plt.ylim(-1, 25)
plt.gca().set_aspect('equal', adjustable='box')

plt.xticks([0,5,10,15,20,25])
plt.yticks([0,5,10,15,20,25])

plt.tick_params(axis='both', direction='in')

plt.grid(True, linewidth=0.5, color='gray',alpha=0.5)

plt.plot([0, 61], [0, 61], color=(17.0/255, 105.0/255, 148.0/255), linewidth=0.5)

plt.text(16, 0.3, 'MAPE=10.24', fontsize=10, ha='left')
plt.text(16, 1.7, 'bias=1.01', fontsize=10, ha='left')
plt.text(16, 3.1, 'MAE=0.56', fontsize=10, ha='left')
plt.text(16, 4.5, 'RMSE=0.95', fontsize=10, ha='left')
plt.text(16, 5.8, 'R²=0.98', fontsize=10, ha='left')
plt.text(16, 7.1, 'N=76', fontsize=10, ha='left')

# plt.legend(loc=(0.035, 0.58), frameon=True, framealpha=0.7, edgecolor='gray',borderaxespad=0.01)

# save tif 420dpi
# plt.savefig(r'D:\iRSDATA\Model_plot\xgbtest\INML\ac\xgb_INMLac20241024.tif', dpi=420)    
plt.show()



