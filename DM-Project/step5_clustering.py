"""
step5_clustering.py
────────────────────
Applies K-Means clustering (k=3) on the training feature set and
visualises how clusters align with the true coral-health labels.

Reads:
  features_train.csv   (produced by step3_feature_extraction.py)

Usage:
    python step5_clustering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load & scale ──────────────────────────────────────────────────────────
    df_train = pd.read_csv("features_train.csv").drop_duplicates().dropna()
    X_train  = df_train.drop("label", axis=1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ── K-Means (k=3: bleached / unbleached / dead) ───────────────────────────
    kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df_train.copy()
    df_clustered["cluster"] = clusters

    print("First 5 rows with cluster assignments:")
    print(df_clustered.head())

    # ── Cluster vs. actual-label cross-table ──────────────────────────────────
    print("\nCluster vs Actual Label:")
    print(pd.crosstab(df_clustered["cluster"], df_clustered["label"]))

    # ── 2-D scatter: avg_B vs avg_G, coloured by cluster, shaped by label ─────
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x=df_clustered["avg_B"],
        y=df_clustered["avg_G"],
        hue=df_clustered["cluster"],
        style=df_clustered["label"],
        palette="Set1",
        s=100,
    )
    plt.title("K-Means Clustering vs Actual Labels (avg_B vs avg_G)")
    plt.xlabel("avg_B")
    plt.ylabel("avg_G")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

    print("\n✅ Clustering analysis complete!")
