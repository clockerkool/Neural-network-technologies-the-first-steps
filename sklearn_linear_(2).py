# -*- coding: utf-8 -*-
"""sklearn_linear (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hw8KCh9MvvmR2OpT6GMKLLPPLnW07pvX

# Линейная регрессия

## Лабораторная работа №2

---

**Впишите в эту ячейку ваши ФИО, группу и вариант**.

ФИО: Султанов Даулет Маратович

Группа: ИТ-0940321

Вариант: 1

---

Далее по ходу ноутбука вам встрется ячейки с кодом, в которых будут комментарии с заданиями, и текстовые ячейки как эта с вопросами, на которые вам необходимо письменно в ноутбуке ответить.

Все ячейки необходимо запускать.

Данные для вариантов лежат по ссылкам:

1. http://labcolor.space/linreg-1.csv
1. http://labcolor.space/linreg-2.csv

Скопируйте ссылку для своего варианта.

### Импортирование модулей для выполнения работы
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 5
from sklearn.linear_model import LinearRegression



df = pd.read_csv("http://labcolor.space/linreg-1.csv")

df.head()
df.describe()


df.isnull() #Пропусков нет

y =  df.pop("y")
X =  df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


regr =  LinearRegression()# ваш код

# Обучение модели с использованием обучающего набора
regr.fit(X_train, y_train)  # ваш код



y_pred = regr.predict(X_test)

print(f"Средняя квадратичная ошибка {mean_squared_error(y_true=y_test, y_pred=y_pred):.2f}")
print(f"Коэффициент детерминации {r2_score(y_true=y_test, y_pred=y_pred):.2f}")


plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue")


df2 = pd.read_csv("http://labcolor.space/linreg-1.csv")

mean = np.mean(df2["y"])
std = np.std(df2["y"])

df2["z-score"] = df2.apply(lambda x: (x['y'] - mean) / std, axis=1)


df2 = df2.loc[(df2['z-score'] <= 3) & (df2['z-score'] >= -3)]
len(df2)

y_smooth = df2.pop("y")
z_score = df2.pop("z-score")
X_smooth = df2

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_smooth, y_smooth, test_size=0.2, random_state = 42)


regr2 = LinearRegression()

# Обучение модели с использованием обучающего набора
regr2.fit(X_train2, y_train2)

# прогноз
y_pred2 = regr2.predict(X_test2)


print(f"Средняя квадратичная ошибка {mean_squared_error(y_test2, y_pred2):.2f}")
print(f"Коэффициент детерминации {r2_score(y_test2, y_pred2):.2f}")




plt.scatter(X_test2, y_test2, color="black")
plt.plot(X_test2, y_pred2, color="blue")


