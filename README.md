
# 📊 Assessing Financial Institutions’ Sectoral Preferences in SME Financing (Malaysia)

This repository presents the analytical framework and machine learning code used to assess sectoral financing preferences of Financial Institutions (FIs) towards Small and Medium Enterprises (SMEs) in Malaysia. The work supports the Master’s research *“Assessing Financial Institutions’ Sectoral Preferences in SME Financing in Malaysia Using Machine Learning and Clustering Analysis”* (2025), conducted using data from Bank Negara Malaysia (BNM) and the Department of Statistics Malaysia (DOSM).

## 🔍 Project Overview

Despite being the engine of Malaysia’s economic growth, SMEs face financing difficulties influenced by institutional biases. This project investigates whether FIs demonstrate preferential treatment toward specific sectors by:

- Analyzing key financial ratios (Approval, Disbursement, Repayment)
- Applying classification models to detect preference patterns
- Scoring sector preferences using model-derived feature importance
- Performing K-means clustering against GDP contributions

## 🧠 Machine Learning Models

Ten supervised classifiers were evaluated to detect FIs’ preference patterns:

- ✅ **Random Forest** (Best model: F1 = 0.1939, AUC-ROC = 0.7002)
- Gradient Boosting
- Decision Tree
- AdaBoost
- Logistic Regression
- k-Nearest Neighbours (k-NN)
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)
- Naïve Bayes
- Linear Model (SGDClassifier)

## 🧪 Methodology

### 1. **Data Sources**
- **Bank Negara Malaysia (BNM)**: SME financing data (Applied, Approved, Disbursed, Repaid) by sector
- **Department of Statistics Malaysia (DOSM)**: Sectoral GDP contributions

### 2. **Feature Engineering**
Derived ratios:
- **Approval Ratio** = Approved / Applied
- **Disbursement Ratio** = Disbursed / Approved
- **Repayment Ratio** = Repaid / Disbursed

### 3. **Label Encoding & Preprocessing**
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode sector
le = LabelEncoder()
y = le.fit_transform(df['Sector'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Approval Ratio', 'Disbursement Ratio', 'Repayment Ratio']])
```

### 4. **Train-Test Split & Stratified Cross Validation**
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 5. **Model Training & Evaluation**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
```

### 6. **Feature Importance & Preference Score**
```python
# Extract feature importance
importance = rf.feature_importances_

# Weighted Preference Score
df['FI_Preference_Score'] = (importance[0] * df['Approval Ratio'] +
                             importance[1] * df['Disbursement Ratio'] +
                             importance[2] * df['Repayment Ratio'])
```

### 7. **Clustering**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['FI_Preference_Score', 'GDP']])
```

## 📈 Visualizations

- Time-series of approved loans by sector
- Feature importance bar plot (Random Forest)
- Elbow plot for optimal cluster count
- Cluster scatter plot (Preference Score vs GDP)

## 📁 Repository Structure

```
.
├── Updated_ML_FIs_Economic_Preferences.ipynb   # Main Jupyter notebook
├── P132415_Final_Report_Using_Machine_Learning.pdf # Full research report
├── data/
│   ├── BNM_SME_Financing.csv
│   └── DOSM_GDP.csv
├── visuals/
│   ├── cluster_plot.png
│   ├── feature_importance.png
├── README.md
```

## 🎓 Citation

If you use this work, please cite:
> Esa, M. W. B. (2025). *Assessing Financial Institutions’ Sectoral Preferences in SME Financing in Malaysia Using Machine Learning and Clustering Analysis*. Universiti Kebangsaan Malaysia.

## 📬 Contact

For questions, reach out to:
- **Author**: Mohammad Wafiuddin bin Esa
- **Email**: [YourEmail@example.com]
- **University**: Universiti Kebangsaan Malaysia
