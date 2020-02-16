
### Dense

link : https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/

```
Dense(8, input_dim=4, init='uniform', activation='relu')
```

* 첫번째 인자 : 출력 뉴런의 수
* input_dim: 입력 뉴런의 수
* init: 가중치 초기화 방법 설정
  * 'uniform': 균일 분포
  * 'normal': 가우시안 분포
* activation: 활성화 함수 설정
  * 'linear' : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
  * 'relu' : rectifier 함수, 은닉층에 주로 쓰임
  * 'sigmoid' : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. (0, 1)
  * 'softmax' : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다. 


### 팁?

* 입력층이 아닐 경우 이전층의 출력 뉴런 수를 알 수 있기 때문에 input_dim을 지정하지 않아도 됩니다.

### 예제들?

* 여러가지 딥러닝이 구현된 예제: https://keras.io/ko/getting-started/sequential-model-guide/
* 코드블록 구현 : https://tykimos.github.io/DeepBrick/