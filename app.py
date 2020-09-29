from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('score.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('trial.html')



@app.route("/score", methods=['POST'])
def score():
    
    
    sentence = request.form['Text']
        
    output=model.score(sentence)
        #output=round(prediction[0],2)
        
    return render_template('index.html',prediction_text=" {}".format(output))
    

if __name__=="__main__":
    app.run(debug=True)