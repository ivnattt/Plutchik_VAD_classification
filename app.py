from flask import Flask, render_template, request, url_for
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('trial.html')


@app.route("/score", methods=['POST'])
#@app.route("/score")
def score():
    
    data_vad=pd.read_excel("VAD.xlsx")
    
    data=pd.read_excel("FIRST_8_EMO.xlsx")
    data.fillna(0, inplace=True)
    datas=pd.get_dummies(data, columns=['emotion'])

    datas['emotion_sadness']=datas['emotion_sadness'].multiply(datas['emotion-intensity-score'])
    datas['emotion_anger']=datas['emotion_anger'].multiply(datas['emotion-intensity-score'])
    datas['emotion_anticipation']=datas['emotion_anticipation'].multiply(datas['emotion-intensity-score'])
    datas['emotion_fear']=datas['emotion_fear'].multiply(datas['emotion-intensity-score'])

    datas['emotion_joy']=datas['emotion_joy'].multiply(datas['emotion-intensity-score'])
    datas['emotion_disgust']=datas['emotion_disgust'].multiply(datas['emotion-intensity-score'])
    datas['emotion_surprise']=datas['emotion_surprise'].multiply(datas['emotion-intensity-score'])
    datas['emotion_trust']=datas['emotion_trust'].multiply(datas['emotion-intensity-score'])

    del datas['emotion-intensity-score']


    def clean_text(text):
        # remove backslash-apostrophe 
        text = re.sub("\'", "", text) 
        # remove everything except alphabets 
        text = re.sub("[^a-zA-Z]"," ",text) 
        # remove whitespaces 
        text = ' '.join(text.split()) 
    
        text = text.lower() 
        return text

    stop_words = set(stopwords.words('english'))

    # function to remove stopwords
    def remove_stopwords(text):
        no_stopword_text = [w for w in text if not w in stop_words]
        return ' '.join(no_stopword_text)


    def score_vad(sentence):
        corpuso=clean_text(sentence)
        corpuso=corpuso.split(' ')
        corpuso=remove_stopwords(corpuso)
        corpuso=corpuso.split(' ')
        return data_vad.loc[data_vad['Word'].isin(corpuso)].sum(axis = 0, skipna = True)[1:]

    def score_emo(sentence):
        corpuso=clean_text(sentence)
        corpuso=corpuso.split(' ')
        corpuso=remove_stopwords(corpuso)
        corpuso=corpuso.split(' ')
        return datas.loc[datas['word'].isin(corpuso)].sum(axis = 0, skipna = True)[1:]
    
    #THE MAIN THING

    if request.method == 'POST':
        message=request.form['Text']
        data= message
        output = score_vad(data)
        output2 = score_emo(data)
        
    return render_template('trial.html',predictionVAD_text=" {} ".format(output), predictionEMO_text=" {} ".format(output2))
    

if __name__=="__main__":
    app.run(debug=True)
