"""
step6_association_rules.py
───────────────────────────
Discretises the training features and mines association rules using
the Apriori algorithm (via mlxtend).

Reads:
  features_train.csv   (produced by step3_feature_extraction.py)

Usage:
    python step6_association_rules.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load training features ─────────────────────────────────────────────────
    df_train = pd.read_csv("features_train.csv").drop_duplicates().dropna()

    # ── Discretise numeric features (High / Low based on median) ──────────────
    df_assoc = df_train.copy()
    df_assoc["brightness_high"] = df_assoc["brightness"] > df_assoc["brightness"].median()
    df_assoc["texture_high"]    = df_assoc["texture"]    > df_assoc["texture"].median()
    df_assoc["coral_dense"]     = df_assoc["coral_count"]> df_assoc["coral_count"].median()

    # Keep only binary/label features and one-hot encode the class label
    assoc_features = ["brightness_high", "texture_high", "coral_dense", "label"]
    df_bin = pd.get_dummies(df_assoc[assoc_features])

    # Make sure all columns are boolean (required by mlxtend)
    df_bin = df_bin.astype(bool)

    print("Binary feature columns:", list(df_bin.columns))
    print(df_bin.head())

    # ── Frequent itemsets (min support = 10 %) ────────────────────────────────
    frequent_itemsets = apriori(df_bin, min_support=0.1, use_colnames=True)
    print(f"\nFrequent itemsets found: {len(frequent_itemsets)}")

    # ── Association rules (min confidence = 60 %) ─────────────────────────────
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    rules_sorted = rules.sort_values("confidence", ascending=False)

    print(f"Association rules found: {len(rules_sorted)}")
    print("\nTop 10 rules by confidence:")
    print(rules_sorted[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))

    # ── Support vs Confidence scatter (bubble size = lift) ────────────────────
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=rules_sorted["support"],
        y=rules_sorted["confidence"],
        size=rules_sorted["lift"],
        hue=rules_sorted["confidence"],
        palette="viridis",
        legend=False,
    )
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules – Support vs Confidence (bubble size = Lift)")
    plt.tight_layout()
    plt.show()

    print("\n✅ Association rule mining complete!")
