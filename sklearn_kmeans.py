import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 5
from sklearn.cluster import KMeans


df = pd.read_csv("http://labcolor.space/kmeans-1.csv")
df.head()
df.describe()


final_data = df.values
kmeans = KMeans(n_clusters=10, init="random", random_state=42)
kmeans.fit(final_data)

print(kmeans.labels_)
print(kmeans.inertia_)



SSE = {}

for k in np.arange(2, 20):
  cluster = KMeans(n_clusters=k, init="random", random_state=42)
  cluster.fit(final_data)
  SSE[k] = cluster.inertia_



lists = SSE.items()
x, y = zip(*lists)

plt.plot(x, y)
plt.show()
print(SSE)


final_kmeans = KMeans(n_clusters=4, init="random", random_state=42)
final_kmeans.fit(final_data)
print(final_kmeans.labels_)

