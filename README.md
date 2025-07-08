# CREDIT-CARD-FRAUD-DETECTION
# Credit Card Fraud Detection 🕵️‍♂️💳

This project demonstrates **credit card fraud detection** using machine learning techniques on an imbalanced dataset. It employs Logistic Regression and Random Forest classifiers with and without resampling techniques like **SMOTE** and **Random Undersampling**.

## 📊 Dataset

- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features: 30 anonymized transaction variables (V1 to V28), `Time`, `Amount`
- Target: `Class` (0 = Genuine, 1 = Fraud)

## 🧠 Models Used

- Logistic Regression
- Random Forest Classifier

## 🧪 Workflow

1. **Import Libraries**
2. **Load Dataset**
3. **Preprocessing**
   - Scaling `Amount` using `StandardScaler`
   - Handling class imbalance
4. **Train/Test Split**
5. **Model Training**
   - Logistic Regression
   - Random Forest
6. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
7. **Resampling Techniques**
   - SMOTE (Oversampling)
   - RandomUnderSampler (Undersampling)

## 🛠️ Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
