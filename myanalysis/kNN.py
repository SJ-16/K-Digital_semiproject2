# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
from config.settings import DATA_DIRS

'''
[Step 1] 데이터 준비
'''

df = pd.read_csv(DATA_DIRS[0]+'//Clothingsizedata.csv');

# 데이터 살펴보기
print(df.head())
print('\n')
print(df.loc[6])
'''
[Step 2] 데이터 탐색
'''

# 데이터 자료형 확인
print(df.info())
print('\n')

'''
[Step 3] 분석에 사용할 속성을 선택
'''

# 분석에 활용할 열(속성)을 선택
ndf = df[['weight', 'age', 'height', 'size']]
print(ndf.head())
print('\n')

# # 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
# onehot_size = pd.get_dummies(ndf['size'])
# ndf = pd.concat([ndf, onehot_size], axis=1)
#
#
#
# ndf.drop(['size'], axis=1, inplace=True)
# print(ndf.head())
# print('\n')
# print(ndf)

'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['weight', 'age', 'height']]  #독립 변수 X
y=ndf['size']       #종속 변수 Y
print('--------------------------------------------22222222222222222222222');
print(X);

# 설명 변수 데이터를 정규화(normalization)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print('--------------------------------------------',X);
# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)


'''
[Step 5] KNN 분류 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 KNN 분류 모형 가져오기
from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성 (k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=228)
# 10 - 0.496/ 50 - 0.5139 / 100 - 0.5170 / 200 - 0.5185195546354837/ 220 - 0.5188272813741398
# / 225 - 0.5198903373804062
# / 227 - 0.520142113802943
# / 228 - 0.5203938902254798 **
# / 230 - 0.5201980641190622
# / 240 - 0.5192748839030941
# / 250 - 0.5189112068483187
# / 300 - 0.5188272813741398/ 400 - 0.5170/ 500 - 0.516 / 1000 - 0.511

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)
print(X_test)
print(knn.predict([[62,28,172.7]]));

# test data를 가지고 y_hat을 예측 (분류)
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.multilabel_confusion_matrix(y_test, y_hat)
print(knn_matrix)

# 모형 성능 평가 - 평가지표 계산
knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)

from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test,y_hat);
print(knn_acc);

#
#
# X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male',
#        'town_C', 'town_Q', 'town_S']]
# data = pd.DataFrame({'pclass':[1],'age':[26],'sibsp':[0],'parch':[0],'female':[0],'male':[1],'town_C':[1],'town_Q':[0],'town_S':[0] })
# #data = pd.DataFrame({'pclass':[3],'age':[32],'sibsp':[0],'parch':[0],'female':[0],'male':[1],'town_C':[0],'town_Q':[1],'town_S':[0] })
# data1 = pd.concat([X,data]);
# print(data1);
# X = preprocessing.StandardScaler().fit(data1).transform(data1)
# y_hat2 = knn.predict(X)
# print(y_hat2)
# print(y_hat2[len(y_hat2)-1])

