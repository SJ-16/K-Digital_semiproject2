# 모델의 성능비교
from matplotlib import pyplot as plt
import seaborn as sns

# plt.style.use("seaborn");

x = [
     "KNN",
     "SVM",
     "Decision Tree",
     "Random Forest",
     "Logistic Regression"
];

y = [
    52.06456666480166,
    51.01040302040320,
    50.82875246527632,
    51.252570181696115,
    51.21739617149894
];

fig, ax = plt.subplots(figsize=(8, 6));
sns.barplot(x=x, y=y, palette="bright");
plt.ylabel("Model Accuracy");
plt.ylim(50,52.5)
plt.title("Model Accuracy Comparison");
plt.show()