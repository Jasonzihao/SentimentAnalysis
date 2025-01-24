# Library imports
import pandas as pd
import numpy as np
import tensorflow as tf
import re

import torch
from numpy import array


from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import nltk

from Analysis import load_model, analysis

nltk.download('stopwords')
print("stopwords down")
from nltk.corpus import stopwords
import io
import json

stopwords_list = set(stopwords.words('english'))
maxlen = 100


# Create the app object
app = Flask(__name__)


# creating function for data cleaning
# from b2_preprocessing_function import CustomPreprocess
# custom = CustomPreprocess()


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    answer = 0
    query_asis = [str(x) for x in request.form.values()]
#     query_list = []
#     query_list.append(query_asis)
    
    # Preprocess review text with earlier defined preprocess_text function
    query_processed_list = []
    for query in query_asis:
        answer = analysis(query)

    if answer > 0.5:
        return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {answer}")
    else:
        return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {answer}")


if __name__ == "__main__":
    app.run(debug=True)
