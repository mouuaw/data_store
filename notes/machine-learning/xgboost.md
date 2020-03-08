# xgboost

참고 링크 

* 조대협의 블로그 : https://bcho.tistory.com/1354
* 브런치? : https://brunch.co.kr/@snobberys/137
  
## XGBoost 란?

XGBoost는 Gradient Boosting 알고리즘을 분산환경에서도 실행할 수 있도록 구현해놓은 라이브러리이다. Regression, Classification 문제를 모두 지원하며, 성능과 자원 효율이 좋아서, 인기있게 사용되는 알고리즘이다.

XGBoost는 여러개의 Decision Tree를 조합해서 사용하는 Ensemble 알고리즘이다.

Ensemble은 여러개의 모델을 조합해서 그 결과를 뽑아 내는 방법이다. 정확도가 높은 강한 모델을 하나 사용하는 것보다, 정확도가 낮은 약한 모델을 여러개 조합하는 방식이 정확도가 높다는 방법에 기반한 방법인데, Bagging과 Boosting 으로 분류된다.

### 장점

1. 병렬 처리를 사용하기에 학습과 분류가 빠르다
2. 유연성이 좋다. 평가 함수를 포함하여 다양한 커스텀 최적화 옵션을 제공한다.
3. Greedy-algorithm 을 사용한 자동 가지치기가 가능하다. 따라서 과적합이 잘 일어나지 않는다
4. 다른 알고리즘과 연계 활용성이 좋다. xgboost 분류기 결론부 아래에 다른 알고리즘을 붙여서 앙상블 학습이 가능하다

## 기본원리

xgboost는 기본적으로 부스팅이라 불리는 기술을 사용한다. 부스팅은 약한 분류기를 세트로 묶어서 정확도를 예측하는 기법이다.

## 하이퍼 파라미터와 XGBoost

XGBoost doc : https://xgboost.readthedocs.io/en/latest/parameter.html

공식 문서에 xgboost를 사용하기 위한 파라미터들이 정리되어 있지만 많이 사용되는 값들을 알아보자