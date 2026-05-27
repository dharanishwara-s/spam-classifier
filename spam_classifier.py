import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]

data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

stemmer = PorterStemmer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [stemmer.stem(word) for word in words
             if word not in stopwords.words('english')]

    return " ".join(words)

data['clean_message'] = data['message'].apply(preprocess_text)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['clean_message'])

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample = ["Congratulations! You won a free iPhone. Click now"]

sample_clean = [preprocess_text(msg) for msg in sample]

sample_vector = vectorizer.transform(sample_clean)

result = model.predict(sample_vector)

if result[0] == 1:
    print("\nThe message is SPAM")
else:
    print("\nThe message is HAM")