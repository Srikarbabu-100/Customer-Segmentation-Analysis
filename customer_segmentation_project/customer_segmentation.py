# ============================================================
#  END-TO-END CUSTOMER SEGMENTATION ANALYSIS
#  Dataset: E-commerce Customer Behavior
# ============================================================
# HOW TO RUN:
#   pip install pandas numpy matplotlib seaborn scikit-learn
#   python customer_segmentation.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d52',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'figure.dpi':       130,
})

PALETTE   = ['#7c3aed', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
ACCENT    = '#7c3aed'
ACCENT2   = '#0ea5e9'

# ─────────────────────────────────────────────────────────────
# STEP 1 · LOAD & UNDERSTAND THE DATASET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — LOAD & UNDERSTAND THE DATASET")
print("="*60)

# ── Change this path to wherever you saved your CSV ──────────
CSV_PATH = "E-commerce_Customer_Behavior_-_Sheet1.csv"
# ─────────────────────────────────────────────────────────────

df_raw = pd.read_csv(CSV_PATH)

print(f"\n📦 Dataset shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\n📋 Column overview:")
print(df_raw.dtypes.to_string())
print("\n🔍 First 5 rows:")
print(df_raw.head().to_string())
print("\n📊 Basic statistics:")
print(df_raw.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2 · DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — DATA CLEANING")
print("="*60)

df = df_raw.copy()

# 2a · Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("\n🔎 Missing values per column:")
print(pd.DataFrame({'count': missing, '%': missing_pct}).query('count > 0'))

# Fill numeric with median (robust to outliers)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical with mode
cat_cols = df.select_dtypes(include='object').columns
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 2b · Duplicates
dupes = df.duplicated().sum()
print(f"\n🔁 Duplicate rows found : {dupes}")
df = df.drop_duplicates()

# 2c · Outlier removal using IQR (only on key numeric columns)
print("\n📐 Outlier treatment (IQR method):")
key_cols = [c for c in ['Purchase Amount (USD)', 'Age', 'Review Rating',
                         'Previous Purchases'] if c in df.columns]
before = len(df)
for col in key_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    removed = ((df[col] < lo) | (df[col] > hi)).sum()
    df = df[(df[col] >= lo) & (df[col] <= hi)]
    print(f"   {col:30s}  removed {removed} outliers")

print(f"\n✅ Rows after cleaning : {len(df)}  (removed {before - len(df)} outliers)")

# ─────────────────────────────────────────────────────────────
# STEP 3 · EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# Identify numeric & categorical columns dynamically
num_features = df.select_dtypes(include='number').columns.tolist()
cat_features = df.select_dtypes(include='object').columns.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("📊 EDA — Numeric Feature Distributions", fontsize=16,
             fontweight='bold', color='white', y=1.01)

for i, col in enumerate(num_features[:6]):
    ax = axes[i // 3][i % 3]
    ax.hist(df[col].dropna(), bins=30, color=PALETTE[i % len(PALETTE)],
            edgecolor='none', alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("eda_distributions.png", bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✅ Saved: eda_distributions.png")

# Correlation heatmap
if len(num_features) >= 2:
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[num_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap='RdYlGn', center=0, ax=ax,
                linewidths=0.4, linecolor='#0f1117',
                annot_kws={'size': 10})
    ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold')
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", bbox_inches='tight',
                facecolor='#0f1117')
    plt.close()
    print("  ✅ Saved: eda_correlation.png")

# Category distributions (top 2 cat columns)
if len(cat_features) >= 1:
    fig, axes = plt.subplots(1, min(2, len(cat_features)),
                             figsize=(16, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, col in zip(axes, cat_features[:2]):
        vc = df[col].value_counts().head(10)
        bars = ax.barh(vc.index[::-1], vc.values[::-1],
                       color=PALETTE[:len(vc)], edgecolor='none')
        ax.set_title(f"Top {col} categories")
        ax.set_xlabel("Count")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("eda_categories.png", bbox_inches='tight',
                facecolor='#0f1117')
    plt.close()
    print("  ✅ Saved: eda_categories.png")

# ─────────────────────────────────────────────────────────────
# STEP 4 · FEATURE SELECTION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — FEATURE SELECTION")
print("="*60)

# We use numeric features directly (adjust column names if yours differ)
SEGMENT_COLS = num_features[:5]   # take first 5 numeric columns
print(f"\n🎯 Features selected for segmentation:")
for c in SEGMENT_COLS:
    print(f"   • {c}")

X = df[SEGMENT_COLS].copy()

# StandardScaler: makes all features same scale (mean=0, std=1)
# Without this, a column like 'Purchase Amount' (0–500) would dominate
# Age (18–70) just because of its larger numbers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n✅ Scaled {X_scaled.shape[1]} features for {X_scaled.shape[0]} customers")

# ─────────────────────────────────────────────────────────────
# STEP 5 · K-MEANS + ELBOW METHOD
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — K-MEANS CLUSTERING (ELBOW METHOD)")
print("="*60)

# Elbow Method: try K = 2..10, record inertia (sum of squared distances)
# The "elbow" point is where adding more clusters stops helping much
inertias    = []
sil_scores  = []
K_range     = range(2, 11)

print("\n  Fitting K-Means for K = 2 … 10 …")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=1000))
    print(f"   K={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.3f}")

# Plot Elbow + Silhouette side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Optimal K Selection", fontsize=16, fontweight='bold', color='white')

# Elbow
ax1.plot(K_range, inertias, 'o-', color=ACCENT, lw=2.5, ms=8,
         markerfacecolor='white', markeredgewidth=2)
ax1.set_title("Elbow Method — Inertia vs K")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.yaxis.grid(True)
ax1.set_axisbelow(True)

# Silhouette
ax2.plot(K_range, sil_scores, 's-', color=ACCENT2, lw=2.5, ms=8,
         markerfacecolor='white', markeredgewidth=2)
ax2.set_title("Silhouette Score vs K  (higher = better)")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)

# Mark best K
best_k = int(K_range[sil_scores.index(max(sil_scores))])
ax2.axvline(best_k, color='#f59e0b', lw=1.5, ls='--')
ax2.text(best_k + 0.15, max(sil_scores), f"Best K={best_k}",
         color='#f59e0b', fontsize=10, va='top')

plt.tight_layout()
plt.savefig("elbow_silhouette.png", bbox_inches='tight', facecolor='#0f1117')
plt.close()
print(f"\n  ✅ Saved: elbow_silhouette.png  |  Best K = {best_k}")

# ─────────────────────────────────────────────────────────────
# STEP 5b · FIT FINAL K-MEANS
# ─────────────────────────────────────────────────────────────
OPTIMAL_K = best_k
km_final  = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

print(f"\n📦 Cluster sizes:")
print(df['Cluster'].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 6 · VISUALIZE CLUSTERS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — CLUSTER VISUALIZATION")
print("="*60)

# Use PCA to compress to 2 dimensions for plotting
# (We can't plot 5 dimensions on a 2D screen — PCA finds the 2 most
#  informative directions)
pca       = PCA(n_components=2, random_state=42)
X_2d      = pca.fit_transform(X_scaled)
var_expl  = pca.explained_variance_ratio_.sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Customer Clusters — PCA 2D Projection", fontsize=16,
             fontweight='bold', color='white')

# Scatter: all clusters
ax = axes[0]
for cid in range(OPTIMAL_K):
    mask = df['Cluster'] == cid
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=PALETTE[cid % len(PALETTE)], label=f"Cluster {cid}",
               s=18, alpha=0.6, edgecolors='none')

# Plot centroids
centroids_2d = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           c='white', marker='*', s=350, zorder=5,
           edgecolors='black', linewidths=0.5, label='Centroid')
ax.set_title(f"PCA Scatter  ({var_expl:.1f}% variance explained)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(fontsize=9)
ax.grid(True)

# Cluster size donut
ax = axes[1]
sizes  = df['Cluster'].value_counts().sort_index()
colors = [PALETTE[i % len(PALETTE)] for i in sizes.index]
wedges, texts, autotexts = ax.pie(
    sizes, labels=[f"Cluster {i}" for i in sizes.index],
    colors=colors, autopct='%1.1f%%', pctdistance=0.78,
    startangle=90, wedgeprops=dict(width=0.55, edgecolor='#0f1117'))
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(10)
ax.set_title("Cluster Size Distribution")

plt.tight_layout()
plt.savefig("cluster_scatter.png", bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  ✅ Saved: cluster_scatter.png")

# ─────────────────────────────────────────────────────────────
# Radar / Spider chart per cluster
# ─────────────────────────────────────────────────────────────
cluster_means  = df.groupby('Cluster')[SEGMENT_COLS].mean()
scaled_means   = (cluster_means - cluster_means.min()) / \
                 (cluster_means.max() - cluster_means.min() + 1e-9)

labels   = SEGMENT_COLS
N        = len(labels)
angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles  += angles[:1]

fig, axes = plt.subplots(1, OPTIMAL_K,
                          figsize=(5 * OPTIMAL_K, 5),
                          subplot_kw=dict(polar=True))
fig.suptitle("Cluster Profiles — Radar Chart", fontsize=16,
             fontweight='bold', color='white')
if OPTIMAL_K == 1:
    axes = [axes]

for cid, ax in enumerate(axes):
    vals = scaled_means.loc[cid].tolist()
    vals += vals[:1]
    color = PALETTE[cid % len(PALETTE)]
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l[:12] for l in labels], size=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=7,
                        color='#8b949e')
    ax.set_facecolor('#1a1d2e')
    ax.set_title(f"Cluster {cid}", color=color, fontsize=12,
                 fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig("cluster_radar.png", bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  ✅ Saved: cluster_radar.png")

# Feature box-plots per cluster
fig, axes = plt.subplots(1, len(SEGMENT_COLS),
                          figsize=(4 * len(SEGMENT_COLS), 6))
fig.suptitle("Feature Distribution per Cluster", fontsize=16,
             fontweight='bold', color='white')

for i, (ax, col) in enumerate(zip(axes, SEGMENT_COLS)):
    groups = [df.loc[df['Cluster'] == c, col].values
              for c in range(OPTIMAL_K)]
    bp = ax.boxplot(groups,
                    patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(color='#8b949e'),
                    capprops=dict(color='#8b949e'),
                    flierprops=dict(marker='o', color='#8b949e',
                                   markersize=3, alpha=0.4))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col[:20])
    ax.set_xlabel("Cluster")
    ax.set_xticklabels([f"C{c}" for c in range(OPTIMAL_K)])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("cluster_boxplots.png", bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  ✅ Saved: cluster_boxplots.png")

# ─────────────────────────────────────────────────────────────
# STEP 7 · CLUSTER PROFILES IN BUSINESS TERMS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 — CLUSTER PROFILES")
print("="*60)

profile = df.groupby('Cluster')[SEGMENT_COLS].agg(['mean', 'median'])
print("\n📊 Cluster mean values (original scale):")
print(cluster_means.round(2).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 8 · BUSINESS RECOMMENDATIONS (printed)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 8 — BUSINESS RECOMMENDATIONS")
print("="*60)
print("""
  General strategies based on typical e-commerce clusters:

  🟣 HIGH-VALUE LOYALISTS (high spend, high frequency)
     → VIP loyalty programme, early access to new products,
       personalised thank-you emails, premium support.

  🔵 POTENTIAL UPSELLS (moderate spend, decent frequency)
     → Bundle offers, tiered discount coupons,
       'Customers also bought…' recommendations.

  🟢 BARGAIN HUNTERS (low spend, discount-driven)
     → Flash sales, clearance emails, free-shipping thresholds
       to nudge cart values up.

  🟡 DORMANT / AT-RISK (low recency, moderate historical spend)
     → Win-back campaigns: 'We miss you!' + 10% off,
       survey to understand churn reason.

  Tailor messaging per cluster — one-size-fits-all marketing
  wastes budget and reduces open-rates.
""")

# ─────────────────────────────────────────────────────────────
# STEP 9 · IMPROVEMENTS — PCA + RFM summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 9 — IMPROVEMENTS & NEXT STEPS")
print("="*60)
print("""
  ① PCA BEFORE CLUSTERING
     Already shown above for visualisation. For clustering you can
     keep 95 % variance:
       pca = PCA(n_components=0.95)
       X_pca = pca.fit_transform(X_scaled)
     Then cluster on X_pca — fewer dimensions = faster, often better.

  ② RFM ANALYSIS
     If your data has order timestamps and order IDs, compute:
       Recency   — days since last purchase
       Frequency — number of orders
       Monetary  — total spend
     These three features alone power excellent segmentation.

  ③ DASHBOARD
     Tools: Streamlit (Python) or Power BI / Tableau.
     Example Streamlit snippet:
       import streamlit as st
       st.write(df.groupby('Cluster').mean())
       st.pyplot(fig)

  ④ OTHER ALGORITHMS TO TRY
     • DBSCAN — finds clusters of arbitrary shape, no need to fix K.
     • Hierarchical / Agglomerative — builds a dendrogram.
     • Gaussian Mixture Models — soft cluster assignments.

  ⑤ AUTOMATED PIPELINE
     Wrap this script into a scheduled job (cron / Airflow) so
     clusters refresh weekly as new orders arrive.
""")

# ─────────────────────────────────────────────────────────────
# STEP 10 · EXPORT RESULTS
# ─────────────────────────────────────────────────────────────
out_csv = "customer_segments.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ Segmented dataset saved → {out_csv}")
print("\n🎉 Analysis complete! Check the PNG files for all charts.\n")
