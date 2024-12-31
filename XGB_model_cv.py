from xgboost import XGBRegressor as XGBR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold


feature = pd.read_excel('feature_new.xlsx')
target = pd.read_excel('target_lg.xlsx')
X = feature
y = target

kf = KFold(n_splits=5, shuffle=True, random_state=420)

r2_scores = []
mae_scores = []
rmse_scores = []
mape_scores = []
bias1_scores = []

predictions = []

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test,y_test)
    
    # seting parameter
    param = {'silent':True
              ,'booster':'gbtree'
              ,"subsample":0.8      
              ,"max_depth":7     
              ,"eta":0.05                              
              #,"gamma":1                                  
              #,"lambda":1                                  
              ,"alpha":0.01                             
              ,"colsample_bytree":0.9
              ,'min_child_weight': 8 
              ,"nfold":5}
    num_round = 500

    watchlist = [(dtrain,'train'), (dtest,'eval')]
    evals_result = {}
    # verbose_eval=False，
    bst = xgb.train(param, dtrain, num_round,watchlist, evals_result=evals_result,early_stopping_rounds=100,verbose_eval=False) 
    
    y_pred = bst.predict(dtest)
    
    y_pred=np.power(10,y_pred)
    y_test=np.power(10,y_test)
    
    y_pred = np.squeeze(y_pred)
    y_test = y_test.values.ravel()
    
    r2 = r2_score(y_test, y_pred)
    mae = MAE(y_test, y_pred)
    rmse = np.sqrt(MSE(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    bias = np.average(np.log10(y_pred) - np.log10(y_test),axis=0)
    bias1 = np.power(10,bias)
    
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    mape_scores.append(mape)
    bias1_scores.append(bias1)

    results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
    predictions.append((y_pred, y_test))
    print(results)

for i in range(len(r2_scores)):
    print(f"Metrics for prediction {i+1}:")
    print("R2:", r2_scores[i])
    print("MAE:", mae_scores[i])
    print("RMSE:", rmse_scores[i])
    print("MAPE:", mape_scores[i])
    print("bias1:", bias1_scores[i])
    print()

avg_r2 = np.mean(r2_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)
avg_mape = np.mean(mape_scores)
avg_bias1= np.mean(bias1_scores)

print("Average Metrics:")
print("Average R2:", avg_r2)
print("Average MAE:", avg_mae)
print("Average RMSE:", avg_rmse)
print("Average MAPE:", avg_mape)    
print("Average bias1:", avg_bias1)  

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

fig = plt.figure(facecolor='white')

colors = ['blue', 'green', 'red', 'gray', (255.0/255, 154.0/255, 34.0/255)]
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']  
edgecolors = ['black', 'black', 'black', 'black', 'black']

# plt.figure(figsize=(6, 6)) 

for i, (y_pred, y_test) in enumerate(predictions):
    color = colors[i] if i < len(predictions) - 1 else colors[-1]
    label = labels[i] if i < len(predictions) - 1 else labels[-1]
    edgecolor= edgecolors[i] if i < len(predictions) - 1 else edgecolors[-1]
    plt.scatter(y_test, y_pred, color=color, label=label,edgecolors=edgecolor, s=45, alpha=0.6)    
    
plt.xlabel('Measured Salinity (ppt)')
plt.ylabel('Estimated Salinity (ppt)')
# plt.title('XGB model')

plt.xlim(-1, 25)
plt.ylim(-1, 25)
plt.gca().set_aspect('equal', adjustable='box')

plt.xticks([0,5,10,15,20,25])
plt.yticks([0,5,10,15,20,25])

plt.tick_params(axis='both', direction='in')
plt.grid(True, linewidth=0.5, color='gray',alpha=0.5)
plt.plot([0, 61], [0, 61], color='black', linewidth=1)

plt.text(16, 0.3, 'MAPE=12.21', fontsize=12, ha='left')
plt.text(16, 1.7, 'bias=0.98', fontsize=12, ha='left')
plt.text(16, 3.1, 'MAE=0.51', fontsize=12, ha='left')
plt.text(16, 4.5, 'RMSE=0.99', fontsize=12, ha='left')
plt.text(16, 5.8, 'R²=0.95', fontsize=12, ha='left')
plt.text(16, 7.1, 'N=229', fontsize=12, ha='left')

plt.legend(loc=(0.035, 0.58), frameon=True, framealpha=0.7, edgecolor='gray',borderaxespad=0.01)

# save TIF with 420dpi
# plt.savefig(r'D:\iRSDATA\Model_plot\xgbtest\INML\ac\xgb_INMLac20240504_Kfold_1.tif', dpi=420)

plt.show()
