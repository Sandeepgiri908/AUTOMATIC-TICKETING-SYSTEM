import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
mymodel = pickle.load(open('finalmodel.pkl', 'rb'))
tfidfmodel=pickle.load(open('vector.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods = ['POST','GET'])
def success():
    if request.method == 'POST' :
        result = request.form['Issue']
        result_pred=mymodel.predict(tfidfmodel.transform([result]))
        return render_template("index.html", y_predict = result_pred)

if __name__=='__main__':
    app.run(debug = True)