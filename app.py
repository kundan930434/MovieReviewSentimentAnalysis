from flask import Flask, render_template, request
import pickle
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

# Preprocessing function
def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [ps.stem(word) for word in text
            if word not in stopwords.words('english')]

    return ' '.join(text)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    review = request.form['review']

    cleaned_review = clean_text(review)

    vector_input = vectorizer.transform([cleaned_review])

    prediction = model.predict(vector_input)[0]

    return render_template(
        'index.html',
        prediction_text=f'Sentiment: {prediction}'
    )

if __name__ == '__main__':
    app.run(debug=True)