# Customer-Segmentation-Analysis
# 🛍️ Customer Segmentation Analysis
### E-commerce Customer Behavior — K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project that segments e-commerce customers into distinct groups using **K-Means Clustering**, enabling targeted marketing strategies.

---

## 📁 Project Structure

```
customer_segmentation_project/
│
├── customer_segmentation.py   # Main analysis script (all 10 steps)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── outputs/
│   └── customer_segments.csv  # Final dataset with Cluster column added
│
├── eda_distributions.png      # Numeric feature histograms
├── eda_correlation.png        # Correlation heatmap
├── eda_categories.png         # Categorical feature bar charts
├── elbow_silhouette.png       # Optimal K selection charts
├── cluster_scatter.png        # PCA 2D cluster scatter + donut
├── cluster_radar.png          # Per-cluster radar/spider charts
└── cluster_boxplots.png       # Feature distributions per cluster
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your CSV file in the project root and update line ~35 in `customer_segmentation.py`:
```python
CSV_PATH = "E-commerce_Customer_Behavior_-_Sheet1.csv"
```

### 4. Run the analysis
```bash
python customer_segmentation.py
```

All charts are saved as PNG files. The segmented dataset is saved to `outputs/customer_segments.csv`.

---

## 📊 What the Script Does

| Step | Description |
|------|-------------|
| 1 | Load & understand dataset (shape, types, stats) |
| 2 | Data cleaning (missing values, duplicates, outliers) |
| 3 | Exploratory Data Analysis with visualizations |
| 4 | Feature selection & StandardScaler normalization |
| 5 | K-Means with Elbow Method + Silhouette Score |
| 6 | Cluster visualization (PCA scatter, radar, boxplots) |
| 7 | Cluster profiling in business terms |
| 8 | Business recommendations per segment |
| 9 | Suggestions: PCA, RFM, DBSCAN, Streamlit dashboard |
| 10 | Export segmented CSV |

---

## 📈 Sample Charts

| EDA Distributions | Elbow Method | Cluster Scatter |
|:-:|:-:|:-:|
| ![](eda_distributions.png) | ![](elbow_silhouette.png) | ![](cluster_scatter.png) |

| Radar Profiles | Box Plots |
|:-:|:-:|
| ![](cluster_radar.png) | ![](cluster_boxplots.png) |

---

## 🧠 Key Concepts

- **K-Means Clustering** — groups customers by minimizing within-cluster distance
- **Elbow Method** — finds optimal K by plotting inertia vs number of clusters
- **Silhouette Score** — measures how well-separated clusters are (-1 to 1, higher = better)
- **StandardScaler** — normalizes features so no single column dominates
- **PCA** — compresses high-dimensional data to 2D for visualization

---

## 💼 Business Segments Found

| Segment | Characteristics | Strategy |
|---------|----------------|----------|
| 🟣 VIP Champions | High spend, high loyalty | Rewards, early access |
| 🔵 Potential Upsells | Mid spend, moderate frequency | Bundles, discount tiers |
| 🟢 Bargain Hunters | Low spend, discount-driven | Flash sales, free shipping |
| 🟡 At-Risk / Dormant | Low recency, past spenders | Win-back campaigns |

---

## 🔧 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 📜 License

MIT License — free to use, modify, and distribute.
