
# ⭐ **DAY 5 — EDA + K-MEANS (Credit Card Customers)**

## 📌 Dataset

`Credit_Card_Customers.csv`
Tumhari Week 1 folder me already hai.

---

# **1️⃣ Load Dataset**

```python
import pandas as pd

df = pd.read_csv("Full_ML_Roadmap/01_Unsupervised_Learning/Credit_Card_Customers.csv")
df.head()
```

---

# **2️⃣ Basic EDA (Simple + Important)**

EDA ka purpose: "Yeh dataset kis type ka hai? Kis cheez ka pattern hai?"

## ✔ Shape

```python
df.shape
```

## ✔ Columns

```python
df.columns
```

## ✔ Missing values

```python
df.isnull().sum()
```

## ✔ Basic statistics

```python
df.describe()
```

**Advice:**
Credit card customers dataset normally clean hota hai.
But categorical columns ko drop/encode karna hota hai.

---

# **3️⃣ Select Numerical Features (Clustering ke liye zaroori)**

DBSCAN ki tarah, yahan bhi sirf numeric features use karenge.

```python
num_cols = [
    'CREDIT_LIMIT',
    'BALANCE',
    'PURCHASES',
    'PAYMENTS',
    'MINIMUM_PAYMENTS',
    'PURCHASES_FREQUENCY'
]

X = df[num_cols]
```

---

# **4️⃣ Handle Missing Values (Important Step)**

```python
X = X.fillna(X.mean())
```

---

# **5️⃣ Scale Data**

K-Means distance-based hai → scaling zaroori.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# **6️⃣ Elbow Method (Find Best K)**

K-Means me humko K choose karna hota hai.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,10), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method - Credit Card Data")
plt.show()
```

**Usually best K = 4 or 5 hota hai.**

---

# **7️⃣ Apply K-Means**

```python
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
```

---

# **8️⃣ Visualize with PCA (2D Plot)**

Because dataset multi-dimensional hai — PCA zaroori.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='viridis', s=60)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (Credit Card Customers) - PCA 2D View")
plt.show()
```

---

# **9️⃣ Cluster Profiling (Most Important Part)**

Yahan se real business insights milte hain.

```python
df.groupby("Cluster")[num_cols].mean()
```

---

# ⭐ **10️⃣ What You’ll Observe (Honest Insights)**

Typical findings:

* **Cluster 0** → Low balance, low credit limit → low-value customers
* **Cluster 1** → High payments, high purchases → premium spenders
* **Cluster 2** → Low payments but high balance → risky customers
* **Cluster 3** → Medium range → average customers

Inko business me "Customer Segmentation" ke liye use karte hain.

---

# ❤️ **Day 5 Summary (Short & Sticky)**

* EDA → missing values, stats
* Only numeric columns for clustering
* Scaling bohot important
* Elbow method → best K choose
* PCA visualization → clean cluster story
* K-Means gives 4–5 meaningful customer groups

---
