import pandas as pd
import re
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

print("Starting training pipeline...")

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
y = df_combined['label_num'].values

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_res, y_train_res)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

joblib.dump(clf, 'logreg_spam_model_tfidf_smote.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

print("Training and saving completed.")
