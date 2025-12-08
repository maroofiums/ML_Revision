

# **Day 1 — Unsupervised Learning (EDA + K-Means)**

## **1️⃣ What is K-Means Clustering?**

* K-Means is an **unsupervised machine learning algorithm** used to **group similar data points into clusters**.
* “**K**” = number of clusters you want to create.
* “**Means**” = the **centroid (mean) of all points in that cluster**.
* So **K-Means** = divide data into **K clusters** and compute **centroids (mean) of each cluster**.

**Idea:**

1. Choose `K` (number of clusters).
2. Randomly initialize `K` centroids.
3. Assign each data point to the **nearest centroid**.
4. Recompute centroids as the **mean of all points in that cluster**.
5. Repeat steps 3–4 until **centroids do not change much** (convergence).

---

## **2️⃣ Why K-Means is called K-Means**

* `K` = Number of clusters
* `Means` = Centroid = mean of points in cluster
* Algorithm **keeps updating mean of points in cluster**, until stable.

---

## **3️⃣ EDA (Exploratory Data Analysis)**

EDA is **first step** before clustering:

* Understand **data distribution**
* Find **missing values**
* Visualize **relationships between features**

Example with **Mall Customers dataset**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Full_ML_Roadmap/01_Unsupervised_Learning/Mall_Customers.csv")

# Preview
print(df.head())

# Check missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Visualize Age vs Annual Income
plt.scatter(df['Age'], df['Annual Income (k$)'])
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income')
plt.show()

# Visualize Spending Score vs Annual Income
plt.scatter(df['Spending Score (1-100)'], df['Annual Income (k$)'])
plt.xlabel('Spending Score')
plt.ylabel('Annual Income')
plt.title('Spending Score vs Annual Income')
plt.show()
```

**EDA Tips:**

* Always **plot pairs of features**
* Check **outliers**
* Helps to **decide K clusters**

---

## **4️⃣ How to Choose K? (Elbow Method)**

**Problem:** How many clusters should we choose?

* Too few clusters → groups are too broad
* Too many clusters → overfitting / meaningless separation

**Elbow Method:**

1. Run K-Means for different `k` values (e.g., 1 to 10)
2. Calculate **Sum of Squared Errors (SSE)** = sum of squared distances of points to their cluster centroid
3. Plot **k vs SSE**
4. Look for **“elbow” point** where SSE reduction slows → choose that `k`

Example:

```python
from sklearn.cluster import KMeans

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # inertia_ = sum of squared distances to centroids

# Plot elbow curve
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters K')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()
```

**Interpretation:**

* Point where SSE reduction slows = **best K**

---

## **5️⃣ K-Means Clustering Code Example**

```python
from sklearn.preprocessing import StandardScaler

# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
k = 5  # assume elbow method suggested 5 clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster info to dataframe
df['Cluster'] = clusters

# Visualize clusters
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('K-Means Clusters')
plt.show()
```

**Notes:**

* `fit_predict` → fits model and returns cluster labels
* Centroids are automatically stored in `kmeans.cluster_centers_`
* Always **scale features** before clustering

---

### ✅ Summary Day 1:

1. **EDA** → Understand data, check missing values, visualize features
2. **K-Means theory** → `K` = clusters, `Means` = centroid
3. **Elbow Method** → Choose optimal K
4. **K-Means code** → Assign clusters, visualize
5. **Tip** → Standardize features before clustering

---

