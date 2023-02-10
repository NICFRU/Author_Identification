from transformers import pipeline, AutoTokenizer
import streamlit as st
import yaml
import pandas as pd
import re
import plotly.express as px
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib

def load_yaml_file(file_path):
    # reads the yml files as a dictionary, were each topic is a key and the values are a list of elements
    with open(file_path, "r", encoding='UTF-8') as stream:
        yaml_dict = yaml.safe_load(stream)
        return yaml_dict


# Huggingface

@st.cache(show_spinner=False)
def text_splitter(text):
    text=   re.sub(" \d+\n", ".", text)
    text=   re.sub("\n\d+", " ", text)
    text=   re.sub("\n", " ", text)
    
    return re.split(r' *[\.\?!][\'"\)\]]* *', text)[:-1]

def num_to_age(n):
    n = int(n)
    if n == 0:
        n = '<15'
    elif n == 1:
        n = '15-19'
    elif n == 2:
        n = '20-24'
    elif n == 3:
        n = '25-29'
    elif n == 4:
        n = '30-34'
    elif n == 5:
        n = '35-39'
    elif n == 6:
        n = '40-44'
    elif n == 7:
        n = '>=45'
    return n

def label_to_age(label):
    if label == 'LABEL_0':
        n = '<15'
    elif label == 'LABEL_1':
        n = '15-19'
    elif label == 'LABEL_2':
        n = '20-24'
    elif label == 'LABEL_3':
        n = '25-29'
    elif label == 'LABEL_4':
        n = '30-34'
    elif label == 'LABEL_5':
        n = '35-39'
    elif label == 'LABEL_6':
        n = '40-44'
    elif label == 'LABEL_7':
        n = '>=45'
    return n

@st.cache(show_spinner=False)
def topic(text):
    topic_list=['Student', 'indUnk', 'Arts', 'Publishing', 'Communications-Media',
       'Education', 'Technology', 'Consulting', 'LawEnforcement-Security',
       'Biotech', 'Government', 'Transportation', 'RealEstate',
       'Internet', 'Chemicals', 'Non-Profit', 'Telecommunications',
       'Museums-Libraries', 'Tourism', 'Engineering', 'InvestmentBanking',
       'Accounting', 'Science', 'BusinessServices', 'Military',
       'Religion', 'Law', 'Fashion', 'Construction', 'Environment',
       'Marketing', 'Manufacturing', 'Banking', 'Advertising',
       'Sports-Recreation', 'Automotive', 'HumanResources',
       'Architecture', 'Agriculture']


    topic = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    preddict = topic(text, topic_list)
    df = pd.DataFrame(preddict).drop("sequence",axis =1)
    df = df.rename({"scores":"Score","labels":"Prediction"},axis="columns")
    fig = px.bar(df, x="Prediction", y="Score",color="Prediction")

    return fig

@st.cache(show_spinner=False)
def age_classifier(text,task):

    if task == 'Regression':
        reg_mod = joblib.load('models/reg_model.joblib')
        y_pred = reg_mod.predict([text])[0]
        return round(y_pred,2)

    elif task == 'Huggingface':
        model = "CoReProg/IntSemAge"
        tokenizer=AutoTokenizer.from_pretrained(model,use_fast=False)
        class_task = pipeline(
        "text-classification", model=model, tokenizer=tokenizer)
        y_pred = class_task(text)[0]['label']
        y_pred = label_to_age(y_pred)

        return y_pred

    elif task == 'Sklearn':
        class_mod = joblib.load('models/class_model.joblib')
        y_pred = class_mod.predict([text])[0]
        y_pred = num_to_age(y_pred)
        return y_pred
    


@st.cache(show_spinner=False)
def pred_gender(text):
    class_mod = joblib.load('models/gender_stacked.joblib')
    y_pred = class_mod.predict([text])[0]
    if y_pred == "1":
        y_pred = "männlich"
    elif y_pred == "2":
        y_pred = "weiblich"
    return y_pred

@st.cache(show_spinner=False)
def pred_stern(text):
    class_mod = joblib.load('models/sign_stacked.joblib')
    y_pred = class_mod.predict([text])[0]
    return y_pred

@st.cache(show_spinner=False)
def hatespeechNLP(text):
    hate_model_path =  "Hate-speech-CNERG/dehatebert-mono-english"
    hate_task = pipeline(
        "text-classification", model=hate_model_path, tokenizer=hate_model_path
    )

    preddict = hate_task(text)[0]
    prob = round(preddict["score"],2)*100
    pred = preddict["label"]

    return prob,pred

@st.cache(show_spinner=False)
def sentimentNLP(text):
    sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer=AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment",use_fast=False)
    sentiment_task = pipeline(
        "sentiment-analysis", model=sentiment_model_path, tokenizer=tokenizer
    )

    preddict = sentiment_task(text)[0]
    prob = round(preddict["score"],2)*100
    pred = preddict["label"]

    return prob, pred

@st.cache(show_spinner=False)
def multi_line_hatespeechNLP(sentlist):
    ergebnis = [hatespeechNLP(sent) for sent in sentlist]
    df = pd.DataFrame(ergebnis, columns = ["Score", "Prediction"])
    fig = px.bar(df, x="Prediction", y="Score",color="Prediction")
    return fig

@st.cache(show_spinner=False)
def multi_line_sentimentNLP(sentlist):
    ergebnis = [sentimentNLP(sent) for sent in sentlist]
    df = pd.DataFrame(ergebnis, columns = ["Score", "Prediction"])
    fig = px.bar(df, x="Prediction", y="Score",color="Prediction")
    return fig



@st.cache(show_spinner=False)
def single_line_hatespeechNLP(text):
    hate_model_path = "Hate-speech-CNERG/dehatebert-mono-english"
    hate_task = pipeline(
        "text-classification", model=hate_model_path, tokenizer=hate_model_path
    )
    preddict = hate_task(text)[0]
    prob = round(preddict["score"]*100,2)
    pred = preddict["label"]
    notprob = 100 - prob
    if pred == "HATE":
        notpred = "NON_HATE"
    elif pred == "NON_HATE":
        notpred = "HATE"
    df = pd.DataFrame.from_dict({"Label":[pred,notpred],"Score":[prob,notprob]})
    fig = px.bar(df, x="Label", y="Score",color="Label")

    return prob, pred, fig


@st.cache(show_spinner=False)
def single_line_sentimentNLP(text):
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)


    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    l = []
    s = []
    for i in range(scores.shape[0]):
        l.append(config.id2label[ranking[i]])
        s.append(round(scores[ranking[i]]*100,2))

    df = pd.DataFrame.from_dict({"Label":l,"Score":s})
    fig = px.bar(df, x="Label", y="Score",color="Label")
    prob = s[0]
    pred = l[0]

    return prob, pred, fig

