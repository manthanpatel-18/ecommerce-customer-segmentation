# 🛒 E-Commerce Customer Segmentation Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
</div>

#### An interactive analytics dashboard that segments 6,000 e-commerce customers using RFM Analysis + K-Means Machine Learning



---

## 📌 Overview

Most businesses treat all their customers the same — same email, same discount, same campaign. This project proves why that's a mistake.

By analyzing **6,000 customers** across **3 years of transaction data ($14.57M revenue)**, this dashboard identifies distinct behavioral segments and reveals that:

- 🔴 Top **20% of customers** drive **65.4% of total revenue**
- 🔴 **50% of new customers** are lost after their very first purchase
- 🔴 The top **3 segments** (31% of customers) generate **78.6% of all revenue**

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **RFM Segmentation** | Rule-based scoring across Recency, Frequency, Monetary dimensions |
| 🤖 **K-Means ML Clustering** | Unsupervised ML finds hidden behavioral groups beyond RFM rules |
| 📈 **Pareto Analysis** | Identifies revenue concentration across customer base |
| 🔄 **Cohort Retention** | Tracks customer retention month-by-month since first purchase |
| 💰 **CLV Estimation** | 3-year Customer Lifetime Value projection per customer |
| 🎛️ **Interactive Filters** | Real-time date range, segment, and revenue filters in sidebar |

---

## 🖥️ Dashboard Preview

> **4 fully interactive tabs - all with real-time sidebar filters**

### Tab 1 - 📊 RFM Segmentation
- Customer distribution donut chart across 8 segments
- Revenue by segment horizontal bar chart
- Full RFM segment summary table
- Recency vs Frequency heatmap (avg monetary value)

### Tab 2 - 🤖 ML Clustering
- K-Means cluster distribution and revenue breakdown
- Cluster profiles table with avg recency, frequency, spend
- 3D scatter plot (Recency × Frequency × Monetary)
- ML vs RFM cross-comparison heatmap
- Elbow method + Silhouette analysis charts

### Tab 3 - 📈 Revenue & Pareto
- Pareto dual-axis chart (top 20% → 65.4% of revenue)
- Monthly revenue trend (Jan 2022 – Dec 2024)
- Revenue breakdown by product category

### Tab 4 - 🔄 Cohort Analysis
- Interactive cohort retention heatmap
- Month-1 / Month-6 / Month-11 retention metrics
- CLV distribution box plots by RFM segment

---

## 📊 Key Findings

```
Total Customers       →   6,000
Total Revenue         →   $14,570,000
Avg Order Value       →   $215.14
Avg Estimated CLV     →   $6,496 (3-year projection)
Date Range            →   Jan 2022 – Dec 2024
Product Categories    →   5 (Electronics, Clothing, Home & Garden, Sports, Books)
RFM Segments          →   8 behavioral segments
ML Clusters           →   8 (K-Means)
Silhouette Score      →   0.386
```

### 🏆 RFM Segment Summary

| Segment | Customers | % Revenue | Action |
|---|---|---|---|
| Champions | 1,136 | 36.5% | Reward & retain |
| Cannot Lose Them | 653 | 21.2% | Urgent re-engagement |
| Loyal Customers | 726 | 20.9% | Loyalty programs |
| Potential Loyalists | 1,067 | 9.7% | Nurture & convert |
| Need Attention | 636 | 3.9% | Reactivation offers |
| At Risk | 328 | 3.1% | Win-back campaigns |
| New Customers | 148 | 0.3% | Onboarding sequence |
| Hibernating | 1,355 | 4.4% | Low-cost re-activation |

---

## 🤖 ML Model Details

- **Algorithm:** K-Means Unsupervised Clustering
- **Features used:** Recency (days), Frequency (count), Monetary (total $) — all StandardScaled
- **K Selection:** Elbow Method + Silhouette Analysis + Davies-Bouldin Index
- **Key ML Insight:** The "Cannot Lose Them" RFM segment splits into **4 distinct ML clusters** — proving ML finds precision that rule-based segmentation cannot

| Metric | Score |
|---|---|
| Silhouette Score | 0.386 |
| Davies-Bouldin Index | 0.817 |
| Calinski-Harabasz Score | 7,645.66 |
| Number of Clusters | 8 |

---

## 🗂️ Project Structure

```
ecommerce-customer-segmentation/
│
├── app.py                          # Main Streamlit dashboard application
├── generate_data.py                # Synthetic data generator (6,000 customers)
├── ml_clustering.ipynb             # Jupyter notebook — K-Means model training
├── requirements.txt                # Python dependencies
├── README.md                       # You are here
│
├── data/
│   └── ecommerce_transactions.csv  # Generated transaction dataset
│
├── models/
│   ├── kmeans_model.pkl            # Trained K-Means model (joblib)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── model_summary.json          # Model metrics and cluster info
│
└── outputs/
    └── ml_results/
        ├── elbow_analysis.png       # Elbow + Silhouette + Davies-Bouldin chart
        ├── 3d_cluster_viz.png       # 3D RFM cluster visualization
        └── ml_vs_rfm_heatmap.png   # ML clusters vs RFM segments comparison
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_data.py
```

### 4. Train the ML model
Open and run all cells in:
```
ml_clustering.ipynb
```
This saves `kmeans_model.pkl`, `scaler.pkl`, and `model_summary.json` to the `models/` folder.

### 5. Run the dashboard
```bash
streamlit run app.py
```

### 6. Open in browser
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
```

Install all at once:
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Pandas** | Data loading, cleaning, RFM aggregation |
| **Scikit-learn** | K-Means clustering, StandardScaler, model metrics |
| **Plotly** | All interactive charts |
| **Streamlit** | Web dashboard framework |
| **Joblib** | Model serialization (save/load pkl) |
| **NumPy** | Numerical operations, CLV calculations |

---

## 💡 Business Insights

1. **Frequency drives value more than Recency** - customers who buy often spend more regardless of when they last bought
2. **Top 20% → 65.4% of revenue** - VIP retention programs have 7× ROI vs mass campaigns
3. **50% Month-1 churn** - biggest revenue leak; fixable with a simple post-purchase onboarding email sequence
4. **ML finds sub-groups within RFM** - "Cannot Lose Them" needs 4 different campaigns, not 1
5. **Electronics = 41.7% of revenue** - category-specific promotions here have outsized returns
6. **Hibernating customers = low ROI** - reallocate budget to Potential Loyalists instead

---

## 📬 Connect With Me

- Linkedin: [Manthan Patel](https://www.linkedin.com/in/manthan-patel18)
- Portfolio: [Manthan Patel](https://manthan-portfolio-opal.vercel.app/)

---

<div align="center">

**⭐ If you found this project useful, please give it a star!**

*Built with Python · Streamlit · Scikit-learn · Plotly*

</div>
