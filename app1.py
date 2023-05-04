from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import ssl
import tensorflow_text as text
import joblib
import sparsh

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

from keras.models import load_model
model = load_model('model(1).h5')
lstm = load_model('lstm.h5')
doc = pickle.load(open('doc.pkl','rb'))
bert = lstm

word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

app = Flask(__name__,static_url_path='/static',template_folder='templates')
@app.route("/")
# Displaying front end
def hello():
    return render_template('index.html')

@app.route("/landing")
def landing():
    return render_template('landing.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/help")
def help():
    return render_template('help.html')

@app.route("/fact")
def fact():
    return render_template('fact.html')


@app.route('/sub', methods=["POST"])

def submit():
    if request.method == 'POST':
        name = request.form['news']
        modelType = request.form['modelType']
    sentence = name
    setence_hindi = sentence
    sentence = sentence.lower()
    punc = list(string.punctuation)
    punc.append('\'')
    punc.append('"')
    print(punc)
    for i in string.punctuation:
        sentence = sentence.replace(i, '')
    tok = sentence.split(' ')
    lemmatizer = WordNetLemmatizer()

    for w in sentence:
        lemmatizer.lemmatize(sentence)

    stop = stopwords.words('english')
    minStop = []
    for i in tok:
        if i not in stop:
            minStop.append(i)
    if modelType == 'dense':
        tag = [TaggedDocument(minStop,[0])]
        predVec = [doc.infer_vector(minStop)]
        predVec = np.array(predVec)
        results = model.predict(predVec)

    elif modelType == 'lstm':
        val = []
        for i in minStop:
            temp = np.array(word2vec([i]))
            val.append(temp)
        val = np.array(val)
        results = lstm.predict(val)

    elif modelType == 'bert':
        val = []
        for i in minStop:
            temp = np.array(word2vec([i]))
            val.append(temp)
        val = np.array(val)
        results = lstm.predict(val)

    elif modelType == 'hindi':
        results_hindi = sparsh.hindi_model([setence_hindi])
        return render_template('fact.html', official = results_hindi)
         

    conVal = results[0][0]
    
    if results[0][0] >= 0.5:
        results = "Fake"
        
    else:
        results = "True"
    return render_template('fact.html', official = results)

if __name__ == "__main__":
    app.run(debug=True,port = 5555)