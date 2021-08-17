# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
from sklearn.metrics import accuracy_score

from config.settings import DATA_DIRS, STATICFILES_DIRS
import time
start = time.time()
'''
[Step 1] 데이터 준비/ 기본 설정
'''

# load_dataset 함수를 사용하여 데이터프레임으로 변환
df = pd.read_csv(DATA_DIRS[0]+'//Clothingsizedata.csv');

'''
[Step 2] 데이터 탐색/ 전처리
'''

'''
[Step 3] 분석에 사용할 속성을 선택
'''
# 분석에 활용할 열(속성)을 선택
ndf = df[['weight', 'age', 'height', 'size']]

'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['weight', 'age', 'height']]  #독립 변수 X
y=ndf['size']       #종속 변수 Y

# 설명 변수 데이터를 정규화(normalization)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')


'''
[Step 5] SVM 분류 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 SVM 분류 모형 가져오기
from sklearn import svm

# 모형 객체 생성 (kernel='rbf' 적용)
svm_model = svm.SVC(kernel='rbf')
#-------하이퍼 파라미터 -------- #
from sklearn.model_selection import GridSearchCV
# parameter 들을 dictionary 형태로 설정
parameters = {'kernel':['rbf'],'C':[0.1,1,10,100,1000],
              'gamma':[0.01,0.1,1,10,100]}
# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정.
### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.
grid_svm_model = GridSearchCV(svm_model, param_grid=parameters, cv=3, refit=True)
# 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가 .
grid_svm_model.fit(X_train, y_train)
# GridSearchCV 결과 추출하여 DataFrame으로 변환
scores_df = pd.DataFrame(grid_svm_model.cv_results_)
print(scores_df);
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
print(scores_df);
print('GridSearchCV 최적 파라미터:', grid_svm_model.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_svm_model.best_score_))
# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
estimator = grid_svm_model.best_estimator_
# GridSearchCV의 best_estimator_는 이미 최적 하이퍼 파라미터로 학습이 됨
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
# ----------------------------- #
second = time.time() - start
print('하이퍼파라미터 실행시간',second)
# train data를 가지고 모형 학습
svm_model.fit(X_train, y_train); # 4분13초 걸림
print(X_test);
print(svm_model.predict([[62,28,172.7]]));
third = time.time() - second
print('모형학습 실행시간',third)
# test data를 가지고 y_hat을 예측 (분류)
y_hat = svm_model.predict(X_test) # 5분27초 걸림
print(y_hat[0:10])
print(y_test.values[0:10])
print('\n')
fourth = time.time() - third
print('y_hat 예측 실행시간',fourth)
# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print(svm_matrix)
print('\n')

# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)
print(svm_report)
finish = time.time() - start
print('총 실행시간',finish);

# Visualising the Training set results
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(1)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = svm_model.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','orange','yellow', 'green','blue','navy','purple'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Size')
plt.legend()
plt.show()
plt.savefig(STATICFILES_DIRS[0]+'//svmtrainscatter.png')
plt.close()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(1)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = svm_model.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','orange','yellow', 'green','blue','navy','purple'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Size')
plt.legend()
plt.show()
plt.savefig(STATICFILES_DIRS[0]+'//svmtestscatter.png')