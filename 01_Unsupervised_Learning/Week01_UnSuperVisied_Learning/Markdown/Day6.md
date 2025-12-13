# Day6 -- Customer Segmentation using Mall Customers

---

## 📌 Summary & Insights — Mall Customers Segmentation

### 🔹 Q&A

#### **1. What customer segments were identified using K-Means clustering?**

K-Means clustering identified **five distinct and interpretable customer segments**:

* **Prudent Older Shoppers (Cluster 0)**
  Older customers (avg. age ≈ 55) with moderate annual income (≈ $47k) and moderate spending behavior (≈ 41).
  → Stable but cautious spenders.

* **High Income, High Spenders (Cluster 1)**
  Young to middle-aged customers (avg. age ≈ 33) with high income (≈ $86k) and very high spending scores (≈ 82).
  → Most valuable customer segment.

* **Young Low Income, High Spenders (Cluster 2)**
  Young adults (avg. age ≈ 26) with low income (≈ $26k) but very high spending (≈ 75).
  → Emotion-driven or lifestyle-oriented spenders.

* **Average Balanced Spenders (Cluster 3)**
  Customers with moderate income (≈ $54k) and moderate spending (≈ 41).
  → Core mass-market segment.

* **High Income, Low Spenders (Cluster 4)**
  Middle-aged to older customers (avg. age ≈ 44) with high income (≈ $90k) but very low spending (≈ 18).
  → High potential but currently under-engaged.

---

#### **2. What segments were identified by DBSCAN (including noise points)?**

DBSCAN identified **six clusters plus a noise group (-1)**:

* **Cluster -1 (Noise / Outliers)**
  Customers with mixed characteristics (avg. age ≈ 40, income ≈ $69k, spending ≈ 32).
  → Do not belong to any dense group (~15% of data).

* **Cluster 0:** Young, low-income, very high spenders

* **Cluster 1:** Middle-aged, low-income, low spenders

* **Cluster 2:** Older customers with average income and moderate spending

* **Cluster 3:** Young adults with average income and balanced spending

* **Cluster 4:** High income, very high spenders

* **Cluster 5:** High income, very low spenders

DBSCAN provided **more granular behavioral separation**, especially for low-density groups.

---

#### **3. How do K-Means and DBSCAN compare?**

Both algorithms identified **similar core customer segments**, but they differ in methodology and outcomes:

* **Outlier Handling:**

  * *K-Means* assigns every customer to a cluster.
  * *DBSCAN* explicitly identifies noise/outliers, which prevents centroid distortion.

* **Cluster Shape & Flexibility:**

  * *K-Means* assumes spherical clusters and requires predefined `K`.
  * *DBSCAN* detects arbitrarily shaped clusters without specifying cluster count.

* **Granularity:**
  DBSCAN revealed additional nuanced groups (e.g., low-income, low-spending middle-aged customers) that were less distinct in K-Means.

---

#### **4. What are the key actionable insights?**

The clustering analysis highlights **clear opportunities for targeted engagement**, while also demonstrating the trade-offs between centroid-based and density-based clustering approaches.

---

## 📊 Data Analysis – Key Findings

* **Consistent High-Value Segment:**
  Both models consistently identified a **“High Income, High Spenders”** segment — the most profitable customer group.

* **Untapped Revenue Potential:**
  The **“High Income, Low Spenders”** group represents customers with strong purchasing power but low engagement.

* **Spending ≠ Income:**
  The presence of **young, low-income, high-spending customers** confirms that spending behavior is not solely income-dependent.

* **DBSCAN’s Strength in Outlier Detection:**
  DBSCAN identified ~15% of customers as noise, revealing irregular or niche behaviors that K-Means cannot isolate.

---

## 🚀 Insights & Recommended Next Steps

* **Target High-Value Customers:**
  Launch loyalty programs, exclusive offers, and premium services for high-income high-spenders.

* **Activate High-Income Low-Spenders:**
  Conduct surveys or qualitative research to understand barriers and design personalized engagement strategies.

* **Investigate Noise Customers:**
  Analyze DBSCAN’s noise group to uncover emerging segments, niche preferences, or data collection gaps.

---

