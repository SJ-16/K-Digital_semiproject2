import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

class Analysis:
    def sizeRecomm(self):

        df = pd.read_csv('C:/project/data/Clothingsizedata.csv');
        print(df);

        print(df.isna().sum());
        print(df.describe());

        # 상관 행렬 살펴보기
        fig, ax = plt.subplots(figsize=(8, 6));
        print(sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis", ));

        df["size"].value_counts();

        plt.style.use("seaborn");
        fig, ax = plt.subplots(figsize=(8, 6));
        sns.countplot(x=df["size"], palette="hls");

        plt.style.use("seaborn");
        fig, ax = plt.subplots(figsize=(8, 6));
        sns.distplot(df["height"], color="r");

        plt.style.use("seaborn");
        fig, ax = plt.subplots(figsize=(8, 6));
        sns.distplot(df["weight"], color="b");

        plt.style.use("seaborn");
        fig, ax = plt.subplots(figsize=(8, 6));
        sns.distplot(df["age"], color="darkorange");

        df["size"].value_counts() # 데이터셋에 숫자 값을 갖도록 옷 사이즈 매핑 XXS:1, S:2, M:3, L:4, XL:5, XXL:6, XXXL:7
        df['size'] = df['size'].map({'XXS': 1, 'S': 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "XXXL": 7});
        print(df.head());

        # 데이터를 훈련 및 테스트 데이터 세트로 분할

        # X data
        X = df.drop("size", axis=1);
        print(X.head());

        # y data
        y = df["size"];
        print(y.head());

        # 데이터를 X 트레인, X 테스트 및 y 트레인, y 테스트로 분할
        from sklearn.model_selection import train_test_split;
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2);

        len(X_train), len(X_test);

        # 선형 회귀 학습 모델
        from sklearn.linear_model import LinearRegression;
        clf = LinearRegression();

        clf.fit(X_train, y_train);

        LinearRegressionScore = clf.score(X_test, y_test);
        print("Accuracy obtained by Linear Regression model:", LinearRegressionScore * 100);

        # Random Forest Classifier
        from sklearn.ensemble import RandomForestClassifier;
        model = RandomForestClassifier();

        model.fit(X_train, y_train);
        model.predict(X_test);
        np.array(y_test);
        RandomForestClassifierScore = model.score(X_test, y_test);
        print("Accuracy obtained by Random Forest Classifier model:", RandomForestClassifierScore * 100);

        # KNeighborsClassifier
        from sklearn.neighbors import KNeighborsClassifier;
        clf1 = KNeighborsClassifier(42);

        clf1.fit(X_train, y_train);
        clf1.predict(X_test);
        np.array(y_test);

        KNeighborsClassifierScore = clf1.score(X_test, y_test);
        print("Accuracy obtained by K Neighbors Classifier model:", KNeighborsClassifierScore * 100);

        # DecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier;
        tree = DecisionTreeClassifier();

        tree.fit(X_train, y_train);
        DecisionTreeClassifierScore = tree.score(X_test, y_test);
        print("Accuracy obtained by Decision Tree Classifier model:", DecisionTreeClassifierScore * 100);


        # 모델의 성능비교
        plt.style.use("seaborn");

        x = ["Random Forest Classifier",
             "K Neighbors Classifier",
             "Decision Tree Classifier",
             "Linear Regression"];

        y = [RandomForestClassifierScore,
             KNeighborsClassifierScore,
             DecisionTreeClassifierScore,
             LinearRegressionScore];

        fig, ax = plt.subplots(figsize=(8, 6));
        sns.barplot(x=x, y=y, palette="crest");
        plt.ylabel("Model Accuracy");
        plt.xticks(rotation=40);
        plt.title("Model Comparison Model Accuracy");

        return 'L'  # 문자열로 반환해야 함

    def Closthes(self):
        import numpy as np;
        import pandas as pd;
        import seaborn as sns;
        import matplotlib.pyplot as plt;
        import warnings;
        warnings.filterwarnings('ignore');

        # 데이터셋 가져오기
        df = pd.read_csv('C:/project/data/Clothingsizedata.csv');

        print(df.shape);
        print(df.head());
        print(df.describe());

        # '체중'과 '신장'의 최소값은 22와 137이고 '나이'의 최소값은 0입니다.
        # Children's Wisconsin의 Normal growth table 'age'가 8 미만인 행을 삭제하겠습니다.
        df = df[df['age'] >= 8];
        print(df.describe());

        print(df.isnull().sum());

        # 높이' 열에 일부 Null 값이 있습니다. Null 값이 있는 행을 삭제하겠습니다.
        df.dropna(inplace=True);
        df.reset_index(inplace=True, drop=True);

        print(df.isnull().sum());
        print(df.shape);

        # EDA(Exploratory Data Ahalysis) 탐색적 데이터 분석
        background_color = '#F8EDF4';
        color_palette = ['#F78904', '#00C73C', '#D2125E', '#693AF9', '#B20600', '#007CDE', '#994936', '#886A00',
                         '#39BBC2'];

        print(df['size'].value_counts());

        # 예상과 달리 7개의 분포를 함께 그리는 것은 어려우며, 구별하기 힘듭니다.
        # 따라서 전체 데이터를 두 부분으로 나눕니다.
        # 'L', 'XL', 'XXL', 'XXXL'의 경우 df_L
        # 'XXS', 'S', 'M'의 경우 df_S

        df_L = df.loc[(df['size'] == 'L') | (df['size'] == 'XL') | (df['size'] == 'XXL') | (df['size'] == 'XXXL')];
        df_S = df.loc[(df['size'] == 'XXS') | (df['size'] == 'S') | (df['size'] == 'M')];

        print(df.plot());

        # Correlation Matrix (상관관계 분석)

        # Label Encoding
        df['size'] = df['size'].map({'XXS': 1, 'S': 2, 'M': 3,
                                     'L': 4, 'XL': 5, 'XXL': 6, 'XXXL': 7});

        print(df.corr());

        f, ax = plt.subplots(1, 1, figsize=(8, 8));

        mask = np.triu(np.ones_like(df.corr()));
        ax.text(1.2, -0.1, 'Correlation Matrix', fontsize=18,
                fontweight='bold', fontfamily='serif');
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu',
                    square=True, mask=mask, linewidth=0.7, ax=ax);

        print(plt.show());

        # Modeling
        from sklearn.linear_model import LogisticRegression;
        from sklearn.ensemble import RandomForestClassifier;
        from sklearn.tree import DecisionTreeClassifier;
        from sklearn.neighbors import KNeighborsClassifier;

        from sklearn.model_selection import train_test_split;
        from sklearn.preprocessing import StandardScaler;
        from sklearn.metrics import accuracy_score;

        models = [];
        scores = [];

        X = df.drop('size', axis=1);
        y = df['size'];

        scaler = StandardScaler();
        X = scaler.fit_transform(X);

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0);

        model = LogisticRegression(solver='saga');
        model.fit(X_train, y_train);
        y_pred = model.predict(X_test);
        score_logreg = accuracy_score(y_test, y_pred);

        print('Accuracy Score of Logistic Regression :', score_logreg);

        models.append('Logistic Regression');
        scores.append(score_logreg);

        model = RandomForestClassifier(max_depth=10);
        model.fit(X_train, y_train);
        y_pred = model.predict(X_test);
        score_rf = accuracy_score(y_test, y_pred);

        print('Accuracy Score of Random Forest Classifier :', score_rf);

        models.append('Random Forest');
        scores.append(score_rf);

        model = DecisionTreeClassifier();
        model.fit(X_train, y_train);
        y_pred = model.predict(X_test);
        score_dt = accuracy_score(y_test, y_pred);

        print('Accuracy Score of DecisionTree Classifier :', score_dt);

        models.append('Decision Tree');
        scores.append(score_dt);

        # 최고 점수를 위해 n_neighbors 찾기
        accuracy = [];

        for i in range(1, 11):
            model = KNeighborsClassifier(n_neighbors=i);
            model.fit(X_train, y_train);
            y_pred = model.predict(X_test);
            accuracy.append(accuracy_score(y_test, y_pred));

        score_knn = max(accuracy);
        print('Accuracy Score of K-Nearest Neighbors Classifier :', score_knn);

        plt.figure(figsize=(10, 5));
        plt.plot(range(1, 11), accuracy, linestyle='dashed', marker='o', color='black',
                 markersize=7, markerfacecolor='red');
        plt.xlabel('n_neighbors');
        plt.ylabel('Accuracy');
        plt.show();

        models.append('K-Nearest Neighbors');
        scores.append(score_knn);

        # 결과
        df_result = pd.DataFrame({'Model': models, 'Score': scores});
        df_result.sort_values(by='Score', ascending=False, inplace=True);
        print(df_result);





if __name__ == '__main__':
    Analysis().sizeRecomm();