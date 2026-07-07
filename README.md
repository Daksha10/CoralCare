# 🪸 CoralCare – Coral Health Classification using Machine Learning

CoralCare is a Machine Learning-based web application that classifies the health status of coral reefs from uploaded images. The system predicts whether a coral is **Bleached**, **Unbleached (Healthy)**, or **Dead** using handcrafted image features and a Random Forest classifier. The project also includes data preprocessing, feature engineering, clustering analysis, and association rule mining for comprehensive data analysis.

---

## 🌟 Features

*  Download coral dataset from Roboflow
*  Automatic dataset preprocessing and cleaning
*  Handcrafted image feature extraction
*  Decision Tree and Random Forest model training
*  Model evaluation using confusion matrices and classification reports
*  Streamlit web application for coral health prediction
*  Prediction confidence visualization
*  K-Means clustering analysis
*  Association Rule Mining using the Apriori algorithm

---

## 🛠 Tech Stack

* Python
* Streamlit
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Roboflow API
* Mlxtend

---

## 📊 Dataset

The dataset is downloaded automatically from **Roboflow** in YOLOv8 format.

The dataset contains three coral health categories:

| Class ID | Coral Health |
| -------- | ------------ |
| 0        | Bleached     |
| 1        | Unbleached   |
| 2        | Dead         |

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Daksha10/CoralCare.git

cd CoralCare
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Project Workflow

Run the following scripts in order.

### Step 1 – Download Dataset

```bash
python step1_download_dataset.py
```

Downloads the coral dataset from Roboflow.

---

### Step 2 – Preprocess Dataset

```bash
python step2_preprocess.py
```

This step:

* Removes corrupt images
* Removes unlabeled images
* Removes empty annotation files

---

### Step 3 – Extract Features

```bash
python step3_feature_extraction.py
```

Extracted features include:

* Average Blue value
* Average Green value
* Average Red value
* Brightness
* Texture
* Coral Count

Feature files generated:

```
features_train.csv
features_valid.csv
features_test.csv
```

---

### Step 4 – Train Models

```bash
python step4_train_models.py
```

Models trained:

* Decision Tree
* Random Forest

Evaluation includes:

* Accuracy
* Classification Report
* Confusion Matrix
* Feature Importance

Saved artifacts:

```
coral_rf_model.pkl
scaler.pkl
label_encoder.pkl
```

---

### Step 5 – K-Means Clustering

```bash
python step5_clustering.py
```

Performs unsupervised clustering to analyze coral health distributions.

Outputs include:

* Cluster assignments
* Scatter plot visualization
* Cluster vs Actual Label comparison

---

### Step 6 – Association Rule Mining

```bash
python step6_association_rules.py
```

Uses the Apriori algorithm to discover relationships among coral features.

Outputs include:

* Frequent Itemsets
* Association Rules
* Support
* Confidence
* Lift

---

### Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## 🖥️ Web Application

The Streamlit application allows users to:

* Upload a coral image
* View extracted image features
* Predict coral health status
* Display prediction probabilities
* View classification results instantly

Predicted classes:

* ⚠️ Bleached
* ☠️ Dead
* ✅ Healthy (Unbleached)

---

## 📈 Machine Learning Pipeline

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Feature Extraction
   │
   ▼
Feature Scaling
   │
   ▼
Random Forest Classifier
   │
   ▼
Prediction
   │
   ▼
Streamlit Web Application
```

---

## 📊 Feature Set

| Feature     | Description                         |
| ----------- | ----------------------------------- |
| avg_B       | Average Blue channel intensity      |
| avg_G       | Average Green channel intensity     |
| avg_R       | Average Red channel intensity       |
| brightness  | Mean image brightness               |
| texture     | Pixel intensity standard deviation  |
| coral_count | Number of annotated coral instances |

---

## 📷 Example Prediction

Input:

```
Uploaded Coral Image
```

Extracted Features:

```
avg_B
avg_G
avg_R
brightness
texture
coral_count
```

Output:

```
Coral Health Status

⚠️ BLEACHED

or

☠️ DEAD

or

✅ HEALTHY (UNBLEACHED)
```

---
