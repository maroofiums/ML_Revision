## 📅 Week 1 — Unsupervised Learning Plan

**Goal:**

* Data samajhna (EDA)
* Clustering karna (K-Means, DBSCAN)
* Dimensionality reduction (PCA)
* Visualization

**Datasets:**

1. [Mall Customers](https://raw.githubusercontent.com/marcopeix/cluster-analysis/master/Mall_Customers.csv)
2. [Credit Card Customers](https://raw.githubusercontent.com/plotly/datasets/master/cc_data.csv)

**Daily Breakdown:**

| Day | Focus         | Dataset               | Task                                                 |
| --- | ------------- | --------------------- | ---------------------------------------------------- |
| 1   | EDA           | Mall Customers        | Load CSV, check missing values, summary stats, plots |
| 2   | K-Means       | Mall Customers        | Cluster customers, visualize clusters                |
| 3   | PCA           | Mall Customers        | Reduce dimensions, visualize 2D clusters             |
| 4   | DBSCAN        | Mall Customers        | Density-based clustering, visualize results          |
| 5   | EDA + K-Means | Credit Card Customers | Segment customers, visualize 2D/3D clusters          |
| 6   | PCA + DBSCAN  | Credit Card Customers | Reduce dimensions + density clustering               |
| 7   | Mini Project  | Both datasets         | Combine techniques, insights report                  |

---

## 💻 Week 1 — Example Code (Mall Customers)

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("Full_ML_Roadmap/01_Unsupervised_Learning/Mall_Customers.csv")

# Basic EDA
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Visualize Age vs Annual Income
plt.scatter(df['Age'], df['Annual Income (k$)'])
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income')
plt.show()

# Standardize features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = clusters

# Plot clusters
plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['KMeans_Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('K-Means Clusters')
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['KMeans_Cluster'], cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Customers')
plt.show()

# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = db_clusters

# Visualize DBSCAN clusters
plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['DBSCAN_Cluster'], cmap='plasma')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('DBSCAN Clusters')
plt.show()
```

---