import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#elbow
# Application de l'algorithme k-Means pour différents nombres de clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Tracé de la courbe de la somme des carrés des distances en fonction du nombre de clusters
plt.plot(range(1, 11), sse)
plt.title("Méthode du coude (elbow)")
plt.xlabel("Nombre de clusters")
plt.ylabel("Somme des carrés des distances intra-classe")
plt.show()