import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("http://labcolor.space/mlp-1.csv", dtype={'100':'category'})
#df.head()
#df.describe()
missing_data = df.isnull()
for column in missing_data.columns.values.tolist(): # Смотрим по каким признакам у нас отсутствующие значения
    print(column)
    print(missing_data[column].value_counts())
    print(" ")

df = df.dropna()


y = df.pop('100')
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled =scaler.transform(X_train)

scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'hidden_layer_sizes': [100, 150, 200],
    'activation': ['identity', 'logistic', 'relu'],
    'learning_rate_init': [0.001, 0.001, 0.001]
}

mlp = MLPClassifier(random_state=42)

grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Гиперпараметры модели с лучшим результатом: {grid_search.best_params_}")
print(f"Лучшая точность при кросс валидации: {grid_search.best_score_:.2f}")

test_accuracy = grid_search.score(X_test_scaled, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.2f}")