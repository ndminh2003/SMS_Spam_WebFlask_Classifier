from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
import pandas as pd
import string
from flask import Flask, render_template, request


def preprocess(text):
    stopword = stopwords.words('english')
    text = text.lower()
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.split()
    text = [i for i in text if i not in stopword]
    wordnet_lem = WordNetLemmatizer()
    text = [wordnet_lem.lemmatize(word) for word in text]
    return text


def prediction(text):
    max_len = 32
    vocab_size = 10000
    x = preprocess(text)
    x = [x]
    token = Tokenizer(num_words=vocab_size)
    token.fit_on_texts(X)
    x_new = token.texts_to_sequences(x)
    x_predict = pad_sequences(x_new, maxlen=max_len, padding='post', truncating='post')
    x_predict = np.asarray(x_predict)
    model = load_model('model')
    result = model.predict(x_predict)
    return result


df = pd.read_csv("final_data.csv",  encoding='utf-8')
X = list(map(preprocess, df['Text']))

# GUI

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global Text
    Text = request.args.get('msg')
    result = prediction(Text)
    result = result[0, 0]
    if result < 0.5:
        return "This SMS is ham"
    else:
        return "This SMS is spam"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=54321)

