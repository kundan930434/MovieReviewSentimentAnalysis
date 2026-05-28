import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing
corpus = []

for review in df['review']:

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review
              if word not in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(corpus)

# Labels
y = df['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save files
pickle.dump(model, open('model.pkl', 'wb'))

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print("Model saved successfully")