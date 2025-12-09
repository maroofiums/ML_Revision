
# **Day 2 — K-Means Clustering (Mall Customers)**

## **1️⃣ Goal**

* Fully understand how **K-Means assigns points to clusters**
* Experiment with **different `k` values**
* Visualize clusters and centroids
* Interpret results → who are your customer segments

---

## **2️⃣ Recap**

* K-Means = divide data into **K clusters**, compute **centroid (mean)**
* **Elbow method** → choose optimal K
* **StandardScaler** → scale features before clustering

---

## **3️⃣ Step-by-Step Workflow**

### Step 1: Load dataset

```python
import pandas as pd

df = pd.read_csv("Full_ML_Roadmap/01_Unsupervised_Learning/Mall_Customers.csv")
print(df.head())
```

---

### Step 2: Select features for clustering

* Usually use **numerical features** for distance-based clustering

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

---

### Step 3: Scale features

* Scaling is important because **K-Means uses Euclidean distance**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### Step 4: Find optimal K (Elbow Method)

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sse = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of clusters K')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()
```

**Tip:** Look for **elbow point** — where SSE reduction slows. Usually 5 for this dataset.

---

### Step 5: Fit K-Means with chosen K

```python
k = 5  # as suggested by elbow
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
df['Cluster'] = clusters
```

---

### Step 6: Visualize clusters with centroids

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis', s=50)
centroids = kmeans.cluster_centers_
# Transform centroids back to original scale for plotting
centroids_original = scaler.inverse_transform(centroids)
plt.scatter(centroids_original[:,0], centroids_original[:,1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering with Centroids')
plt.legend()
plt.show()
```

**Explanation:**

* Colored points → cluster assignment
* Red X → centroid of each cluster
* Helps **interpret clusters** → e.g., high income + high spending = premium customers

---

### Step 7: Analyze clusters

```python
# Group by cluster
cluster_summary = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()
print(cluster_summary)
```

**Tip:**

* Analyze clusters → gives **insights for marketing / segmentation**
* Example: cluster with **young + high spending** = target group

---

### ✅ **Day 2 Summary**

1. Select features and scale them
2. Use **Elbow method** to find K
3. Fit **K-Means** → assign clusters
4. Visualize clusters + centroids
5. Analyze cluster statistics → **insights**
