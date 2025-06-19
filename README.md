# 🛡️ AdversaTF-IDF Spam Detector

A **robust SMS spam detection** system using TF-IDF, adversarial obfuscation, SMOTE oversampling, and logistic regression for reliable classification.

---

## ✨ Features

- 🧹 **Text Cleaning:** Lowercase & remove non-alphabetic characters for consistent input  
- 🕵️‍♂️ **Adversarial Obfuscation:** Simulates spammer tricks by replacing characters & adding random punctuation to boost robustness  
- 📊 **TF-IDF Vectorization:** Uni- and bi-gram text features (max 5000) to capture message semantics  
- ⚖️ **SMOTE Oversampling:** Balances spam/ham classes by synthetically oversampling the minority class  
- 🤖 **Logistic Regression:** Fast, interpretable classifier trained on balanced, augmented data  
- 📈 **Evaluation:** Detailed metrics including precision, recall, F1-score, accuracy (~98%), and ROC AUC (~0.99)  

---

## 📂 Dataset

The model is trained and evaluated on the **UCI SMS Spam Collection Dataset**, which contains 5,572 labeled SMS messages.

---

## 🚀 Performance

- **Accuracy:** 98%  
- **Spam Recall:** ~93%  
- **Spam Precision:** ~91%  
- **ROC AUC:** 0.99  

---

## 💾 Saving & Loading

The trained model and TF-IDF vectorizer are saved as:

- `logreg_spam_model_tfidf_smote.joblib`  
- `tfidf_vectorizer.joblib`

---

## ⚙️ Usage

1. Run the training script to train and save the model  
2. Use the evaluation script to load saved models and test on combined clean + adversarial datasets

---

## 🛠️ Requirements

- Python 3.x  
- pandas  
- scikit-learn  
- imblearn (for SMOTE)  
- joblib  

---
