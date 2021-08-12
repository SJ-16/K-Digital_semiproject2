# -*- coding: utf-8 -*-

# 기본 라이브러리 불러오기
import warnings

from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from config.settings import DATA_DIRS

'''
[Step 1] 데이터 준비/ 기본 설정
'''

df = pd.read_csv(DATA_DIRS[0]+'//sizedata.csv');

# 열 이름 지정
df.columns=['weight','age','height','size'];

'''
[Step 2] 데이터 탐색
'''

# 데이터 살펴보기
print(df.head())
print('\n')

# 데이터 자료형 확인
print(df.info())
print('\n')

# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

df.dropna(axis=0, subset=['age','height'],inplace=True); # na값 제거
df['age'] = df['age'].astype('int')       # 문자열을 정수형으로 변환

print(df.describe())                                      # 데이터 통계 요약정보 확인
print('\n')


'''
[Step 3] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X = df[['weight', 'age', 'height']]  # 설명 변수 X
y = df['size']  # 예측 변수 Y

# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')


'''
[Step 4] Decision Tree 분류 모형 - sklearn 사용
'''

# DT 모형 성능 평가 - 평가지표 계산
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree_model.fit(X_train, y_train)
y_hat = tree_model.predict(X_test)      # 2: benign(양성), 4: malignant(악성)
tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)
knn_acc = accuracy_score(y_test,y_hat);
print('DT',knn_acc);


# SVM 모형 성능 평가 - 평가지표 계산
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_hat = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test,y_hat);
print('SVM',svm_acc);

# KNN모형 성능 평가 - 평가지표 계산
knn_model = KNeighborsClassifier(n_neighbors=228)
knn_model.fit(X_train, y_train)
y_hat = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test,y_hat);
print('KNN',knn_acc);

# 앙상블모델 1 - voting 모형 성능 평가 - 평가지표 계산

hvc = VotingClassifier(estimators=[('KNN',knn_model),
                                   ('SVM',svm_model),
                                   ('DT',tree_model)],voting='hard')
hvc.fit(X_train,y_train);
y_hat = hvc.predict(X_test);
hvc_acc = accuracy_score(y_test,y_hat);
print('HVC',hvc_acc);

# 앙상블모델 2 -배깅 (random forest) 모형 성능 평가 - 평가지표 계산
rfc = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=10);
rfc.fit(X_train, y_train);
y_hat = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test,y_hat);
print('RFC',rfc_acc);

# 앙상블모델 3 -  Boosting 모형 성능 평가 - 평가지표 계산
warnings.filterwarnings("ignore")
xgbc = XGBClassifier(n_estimators=50,max_depth=5,random_state=10);
xgbc.fit(X_train, y_train);
y_hat = xgbc.predict(X_test)
xgbc_acc = accuracy_score(y_test,y_hat);
print('XGC',xgbc_acc);