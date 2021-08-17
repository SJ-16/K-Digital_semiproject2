# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
import seaborn as sns
from config.settings import DATA_DIRS, STATICFILES_DIRS
#----------------------plot 라이브러리 ----------------------------------
import matplotlib
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font) ## 그래프를 알아보기위해 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False # 음수(-)값도 깨짐 방지
#-----------------------------------------------------------------------

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
# Label Encoding
df['size'] = df['size'].map({'XXS': 1, 'S': 2, 'M' : 3,
                             'L' : 4, 'XL' : 5, 'XXL' : 6, 'XXXL' : 7})

# 분석에 활용할 열(속성)을 선택
ndf = df[['weight', 'age', 'height', 'size']]
print(ndf.head())
print('\n')

# # 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
# onehot_size = pd.get_dummies(ndf['size'])
# ndf = pd.concat([ndf, onehot_size], axis=1)
# ndf.drop(['size'], axis=1, inplace=True)
# print(ndf.head())
# print('\n')
# print(ndf)


# # Data Heatmap
# ndf_corr = ndf.corr()
# sb.heatmap(ndf_corr, cmap = 'Blues', annot = True, xticklabels = ndf_corr.columns.values, yticklabels = ndf_corr.columns.values)
# plt.title('Data Heatmap', fontsize = 15)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.savefig(STATICFILES_DIRS[0]+'//heatmap.png')
'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['weight', 'age', 'height']]  #독립 변수 X
y=ndf['size']       #종속 변수 Y
print('--------------------------------------------22222222222222222222222');
print(X);
#--------------------------------------------
# scatter visualization
# sb.scatterplot('weight', 'age', data = ndf, hue = 'size', palette = 'Set2', edgecolor = 'b', s = 150,
#                alpha = 0.7)
# plt.title('weight / age')
# plt.xlabel('weight')
# plt.ylabel('age')
# plt.legend(loc = 'upper left', fontsize = 12)
# plt.savefig(STATICFILES_DIRS[0]+'//weight_age.png')
# import seaborn as sb
# sb.scatterplot('weight', 'height', data = ndf, hue = 'size', palette = 'Set2', edgecolor = 'b', s = 150,
#                alpha = 0.7)
# plt.title('weight / height')
# plt.xlabel('weight')
# plt.ylabel('height')
# plt.legend(loc = 'upper left', fontsize = 12)
# plt.savefig(STATICFILES_DIRS[0]+'//weight_height.png')
# import seaborn as sb
# sb.scatterplot('height', 'age', data = ndf, hue = 'size', palette = 'Set2', edgecolor = 'b', s = 150,
#                alpha = 0.7)
# plt.title('height / age')
# plt.xlabel('height')
# plt.ylabel('age')
# plt.legend(loc = 'upper left', fontsize = 12)
# plt.savefig(STATICFILES_DIRS[0]+'//height_age.png')

# # Scatter Matrix
# sb.pairplot(data = ndf, hue = 'size',palette = 'magma')
# plt.savefig(STATICFILES_DIRS[0]+'//pairplot.png')
# plt.show();
#----------------------------------------
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

# Visualising the Training set results
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.02),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.02))
# Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(1)]).T
# # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
# pred = knn.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
# plt.contourf(X1, X2, pred, cmap = ListedColormap(('red', 'green')) )
#
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red','orange','yellow', 'green','blue','navy','purple'))(i), label = j)
# plt.title('KNN (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Size')
# plt.legend()
# plt.savefig(STATICFILES_DIRS[0]+'//knntrainscatter.png')

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.02),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.02))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(1)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = knn.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','orange','yellow', 'green','blue','navy','purple'))(i), label = j)
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Size')
plt.legend()
plt.savefig(STATICFILES_DIRS[0]+'//knntestscatter.png')

# X=ndf[['weight', 'age', 'height']]
# data = pd.DataFrame({'weight':[59],'age':[36],'height':[167.6]})
# #data = pd.DataFrame({'weight':[3],'age':[32],'height':[0],'size':[0]})
# data1 = pd.concat([X,data]);
# print(data1);
# X = preprocessing.StandardScaler().fit(data1).transform(data1)
# y_hat2 = knn.predict(X)
# print(y_hat2)
# print(y_hat2[len(y_hat2)-1])