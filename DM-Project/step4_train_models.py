"""
step4_train_models.py
──────────────────────
Trains a Decision Tree and a Random Forest on the extracted features.

Reads:
  features_train.csv
  features_valid.csv
  features_test.csv   (produced by step3_feature_extraction.py)

Saves:
  coral_rf_model.pkl
  scaler.pkl
  label_encoder.pkl

Usage:
    python step4_train_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_split(csv_path: str):
    df = pd.read_csv(csv_path).drop_duplicates().dropna()
    X  = df.drop("label", axis=1)
    y  = df["label"]
    return X, y


def plot_confusion_matrix(cm, classes, title: str, cmap: str = "Blues") -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, title: str) -> None:
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n{title}:\n")
    print(fi.to_string(index=False))

    plt.figure(figsize=(8, 4))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load CSVs ──────────────────────────────────────────────────────────────
    print("Loading feature CSVs …")
    X_train, y_train = load_split("features_train.csv")
    X_valid, y_valid = load_split("features_valid.csv")
    X_test,  y_test  = load_split("features_test.csv")

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)
    y_test_enc  = le.transform(y_test)
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # ── Scale features ─────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled  = scaler.transform(X_test)

    # ═══════════════════════════════════════════════════════════════════════════
    # 1.  Decision Tree
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Decision Tree ──")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_scaled, y_train_enc)

    y_val_pred_dt = dt.predict(X_valid_scaled)
    val_acc_dt    = accuracy_score(y_valid_enc, y_val_pred_dt)
    print(f"Validation Accuracy : {val_acc_dt * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_valid_enc, y_val_pred_dt, target_names=le.classes_))

    cm_dt = confusion_matrix(y_valid_enc, y_val_pred_dt)
    plot_confusion_matrix(cm_dt, le.classes_,
                          "Decision Tree – Validation Confusion Matrix", "Blues")
    plot_feature_importance(dt, X_train.columns, "Decision Tree Feature Importance")

    # Visualise the tree (requires graphviz system package)
    try:
        import graphviz
        dot_data = export_graphviz(
            dt, out_file=None,
            feature_names=X_train.columns,
            class_names=le.classes_,
            filled=True, rounded=True, special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree_coral", format="pdf", cleanup=True)
        print("Decision tree saved to decision_tree_coral.pdf")
    except ImportError:
        print("graphviz not installed – skipping tree visualisation.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2.  Random Forest
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Random Forest ──")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train_enc)
    print("Training complete!")

    # Validation
    y_val_pred_rf = rf.predict(X_valid_scaled)
    val_acc_rf    = accuracy_score(y_valid_enc, y_val_pred_rf)
    print(f"Validation Accuracy : {val_acc_rf * 100:.2f}%")
    print("\nValidation Classification Report:\n")
    print(classification_report(y_valid_enc, y_val_pred_rf, target_names=le.classes_))

    cm_val_rf = confusion_matrix(y_valid_enc, y_val_pred_rf)
    plot_confusion_matrix(cm_val_rf, le.classes_,
                          "Random Forest – Validation Confusion Matrix", "Greens")

    # Test
    y_test_pred_rf = rf.predict(X_test_scaled)
    test_acc_rf    = accuracy_score(y_test_enc, y_test_pred_rf)
    print(f"Test Accuracy : {test_acc_rf * 100:.2f}%")
    print("\nTest Classification Report:\n")
    print(classification_report(y_test_enc, y_test_pred_rf, target_names=le.classes_))

    cm_test_rf = confusion_matrix(y_test_enc, y_test_pred_rf)
    plot_confusion_matrix(cm_test_rf, le.classes_,
                          "Random Forest – Test Confusion Matrix", "Oranges")

    plot_feature_importance(rf, X_train.columns, "Random Forest Feature Importance")

    # Visualise one tree from the forest
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[0],
              feature_names=X_train.columns,
              class_names=le.classes_,
              filled=True, rounded=True, fontsize=12)
    plt.title("Random Forest – First Tree Visualisation")
    plt.tight_layout()
    plt.show()

    # ─── Save model artefacts ──────────────────────────────────────────────────
    joblib.dump(rf,     MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    print(f"\n✅ Model artefacts saved: {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")
