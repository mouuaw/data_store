# https://dacon.io/competitions/official/235608/codeshare/1112?page=1&dtype=recent&ptype=pub
# %%
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')
jupyter notebook 깔고 ipynb확장자로 저장하면 됨
# %% 
os.chdir('C:/Users/user1/Desktop/study_python')
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submission= pd.read_csv('./sample_submission.csv')

# %%
train.shape, test.shape, submission.shape
# %%
train.isnull().sum().plot()
pob
# %%
for i in range(train.shape[1]):
    if i<=71:
        print( train.columns[i].ljust(10, " "), 'train : ', str(train.isnull().sum()[i]).rjust(6," "), '    test : ', str(test.isnull().sum()[i]).rjust(6, " "))
    else:
        print( train.columns[i].ljust(10, " "),  'train : ',str(train.isnull().sum()[i]).rjust(6," "))
# %%
from sklearn.metrics import mean_absolute_error # 평가지표인 MAE 정의
from sklearn.impute import KNNImputer # KNN 방식을 통한 Imputation
# %%
imputer = KNNImputer(n_neighbors=3)
tr_filled = imputer.fit_transform(train)
ts_filled = imputer.fit_transform(test)
tr_filled = pd.DataFrame(tr_filled, columns=train.columns)
ts_filled = pd.DataFrame(ts_filled, columns=test.columns)


# %%
import xgboost as xgb                       # XGBoost 패키지
from sklearn.model_selection import KFold   # K-Fold CV

# XGBoost : 트리 기반의 앙상블, 캐글 경연 대회에서 상위를 차지한 많은  Data Scientist 가 활용하는 알고리즘으로 유명해짐
# 주요 장점 : GBM 대비 시간이 빠름, 뛰어난 예측 성능, 과적합 규제 등
# %%
def train_model(x_data, y_data, k=10):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
        d_train = xgb.DMatrix(data = x_train, label = y_train)
        d_val = xgb.DMatrix(data = x_val, label = y_val)
        
        wlist = [(d_train, 'train'), (d_val, 'eval')]
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'seed':777
            }

        model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
        models.append(model)
    
    return models

    models = {}

for label in y_train.columns:
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[label])

for col in models:
    preds = []
    for model in models[col]:
        preds.append(model.predict(xgb.DMatrix(test.loc[:, '650_dst':])))
    pred = np.mean(preds, axis=0)

    submission[col] = pred