import pandas as pd
from config.settings import DATA_DIRS, STATICFILES_DIRS
import pickle
from sklearn import preprocessing

class Analysis:
    def sizeRecomm(self,age,height,weight):
        ## Load pickle
        with open(DATA_DIRS[0]+"//data.pickle", "rb") as file:
            ndf = pickle.load(file)
        with open(DATA_DIRS[0]+"//knn.pickle", "rb") as file:
            knn = pickle.load(file)
        X = ndf[['weight', 'age', 'height']]
        data = pd.DataFrame({'weight': [weight], 'age': [age], 'height': [height]})
        data1 = pd.concat([X, data]);
        X = preprocessing.StandardScaler().fit(data1).transform(data1);
        y_hat2 = knn.predict(X);
        result = y_hat2[len(y_hat2) - 1];
        return result;

if __name__ == '__main__':
    Analysis().sizeRecomm(21,120,10);



