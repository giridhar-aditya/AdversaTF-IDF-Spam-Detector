import pandas as pd
import re
import random
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

print("Starting evaluation...")

# Load model and vectorizer
clf = joblib.load('logreg_spam_model_tfidf_smote.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['message'].apply(clean_text)

def obfuscate(text):
    substitutions = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 'u': 'ü', 's': '$'}
    text = ''.join(substitutions.get(c, c) for c in text)
    if len(text) > 0:
        positions = sorted(random.sample(range(len(text)), k=max(1, int(len(text) * 0.05))))
        for p in reversed(positions):
            text = text[:p] + random.choice(['.', '-', '*', '~']) + text[p:]
    return text

df_adv = df.copy()
df_adv['cleaned'] = df_adv['cleaned'].apply(obfuscate)

df_combined = pd.concat([
    df.sample(frac=0.5, random_state=42),
    df_adv.sample(frac=0.5, random_state=42)
], ignore_index=True)

X_text = df_combined['cleaned'].tolist()
y_true = df_combined['label_num'].values

X_tfidf = tfidf.transform(X_text)

y_pred = clf.predict(X_tfidf)
y_proba = clf.predict_proba(X_tfidf)[:, 1]

print("\nClassification Report:\n", classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_true, y_proba):.4f}")

print("Evaluation completed.")
