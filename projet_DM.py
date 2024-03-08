import pandas as pd
from sklearn.preprocessing import StandardScaler
# Lecture du fichier et stockage des données dans un dataframe pandas
fromage_df = pd.read_excel("Fromage.xlsx")

# Suppression de la colonne "Fromages" qui ne contient que des chaînes de caractères
fromage_df = fromage_df.drop(columns=["Fromages"])

# Normalisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(fromage_df)

#la classification ascendante hiérarchique (CAH) à l'aide du package SciPy 

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Appliquer la méthode de liaison de Ward pour mesurer la distance entre les clusters
Z = linkage(fromage_df, method="ward", metric="euclidean")

# Tracer le dendrogramme
fig = plt.figure(figsize=(25, 10))
dendrogram(Z, labels=fromage_df.index, leaf_font_size=10)
plt.show()

#la méthode des centres mobiles (k-Means) à l'aide du package Scikit-Learn :
    
from sklearn.cluster import KMeans

# Créer une instance de l'algorithme k-Means
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)

# Exécuter l'algorithme sur les données de fromage_df
kmeans.fit(fromage_df)

# Afficher les centres de chaque cluster
print(kmeans.cluster_centers_)
# Affichage des résultats de K-means
plt.scatter(fromage_df.iloc[:,2], fromage_df.iloc[:,3], c=kmeans.labels_, cmap='rainbow')

