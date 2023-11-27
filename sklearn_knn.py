import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 5

df = pd.read_csv("http://labcolor.space/knn-1.csv", dtype={'target':'category'})
print(df.describe(include='number'))
print(df.describe(include='category'))


labels, counts = np.unique(df['target'], return_counts=True)
fig, axs = plt.subplots(ncols=1)
sns.barplot(x=[str(label) for label in labels], y=counts, ax=axs).set_title("Распределение по классам")
plt.show()

y = df.pop('target')
X = df



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)


labels, counts = np.unique(y_test, return_counts=True)
fig, axs = plt.subplots(ncols=1)
sns.barplot(x=[str(label) for label in labels], y=counts, ax=axs).set_title("Распределение по классам в тестовой выборке")
plt.show()


scaler = StandardScaler()

scaler.fit(X_train)
scale_scaler_train = scaler.transform(X_train)

scaler.fit(X_test)
scale_scaler_test = scaler.transform(X_test)


clf = KNeighborsClassifier(10, metric="euclidean", algorithm='brute')
clf.fit(scale_scaler_train, y_train)
clf.score(scale_scaler_test, y=y_test)

disp = ConfusionMatrixDisplay.from_estimator(
    clf,
    scale_scaler_test,
    y_test,
    display_labels=np.unique(y),
    cmap=plt.cm.Blues,
    normalize=None,
)
disp.plot()
plt.show()
