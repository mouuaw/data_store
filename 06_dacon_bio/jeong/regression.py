
# LinearRegression : 예측값과 실체값의 RSS (Residaul Sum of Sqaures)를 최소화해 OLS(Ordinary Least Sqaures) 추정 방식으로 구현한 클래스
# 1. MAE (Mean Absoulte Error) : 실제값과 예측값의 차이를 절댓값으로 변환해 평균한 것
# 2. MSE (Mean Sqaured Error) : 실제값과 예측값의 차이를 제곱해 평균한 것
# 3. RMSE (Root Mean Squared Error) : MSE 값은 오류의 제곱을 구하므로 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 씌운 것

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
% matplotlib inline

boston = load_Boston()
boston = pd.DataFrame(boston.data, columns=boston.feature_names) # DataFrame 변환
boston["PRICE"] = boston.target
boston.shape


