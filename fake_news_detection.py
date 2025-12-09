# -*- coding: utf-8 -*-
"""Fake_news_detection.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# ------------------------------------------------------------
# NLP Tools
# ------------------------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ------------------------------------------------------------
# TEXT CLEANING FUNCTIONS
# ------------------------------------------------------------
def remove_noise(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # KEEP numbers
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF]+', ' ', text)
    return text

def normalization(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize(text):
    return word_tokenize(text)

def stop_word_removal(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def lemmatizing(tokens):
    le = WordNetLemmatizer()
    return [le.lemmatize(t) for t in tokens]

def txt_prep(text):
    text = remove_noise(text)
    text = normalization(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = stop_word_removal(tokens)
    tokens = lemmatizing(tokens)
    return " ".join(tokens)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
drive.mount('/content/drive')

df_true = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data for projects/true.csv')
df_fake = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data for projects/fake.csv')

# ✔ FIX LABELS (industry standard)
df_true["label"] = 1   # TRUE
df_fake["label"] = 0   # FAKE

df = pd.concat([df_true, df_fake], ignore_index=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ------------------------------------------------------------
# APPLY CLEANING
# ------------------------------------------------------------
df["cleaned_text"] = df["text"].apply(txt_prep)
df["cleaned_title"] = df["title"].apply(txt_prep)
df["cleaned_subject"] = df["subject"].apply(txt_prep)

df_cleaned = df[["cleaned_text","cleaned_subject","cleaned_title","date","label"]].copy()
df_cleaned["label_t"] = df_cleaned["label"].map({1:"true", 0:"fake"})

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(8,8))
sns.countplot(
    data=df_cleaned,
    y="cleaned_subject",
    hue="label_t",
    order=df_cleaned["cleaned_subject"].value_counts().index
)
plt.title("Number of True vs Fake News by Subject")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# BUILD FULL TEXT FOR TRAINING
# ------------------------------------------------------------
df_cleaned["full_text"] = (
    df_cleaned["cleaned_title"] + " " +
    df_cleaned["cleaned_subject"] + " " +
    df_cleaned["cleaned_text"]
)

X = df_cleaned["full_text"]
y = df_cleaned["label"]

# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# IMPROVED TF-IDF
# ------------------------------------------------------------
tf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1,2)
)

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)

# ------------------------------------------------------------
# IMPROVED LOGISTIC REGRESSION
# ------------------------------------------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"   # ✔ FIX
)

model.fit(X_train_tf, y_train)
y_pred = model.predict(X_test_tf)

print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"])
disp.plot(cmap="Oranges", values_format='d')
plt.title("Confusion Matrix")
plt.show()

# ------------------------------------------------------------
# SAVE MODEL + VECTORIZER
# ------------------------------------------------------------
joblib.dump(model, 'my_Fake_news_detection_model.pkl')
joblib.dump(tf, 'my_vectorizer.pkl')