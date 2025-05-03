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

# nltk.download('stopwords')
# print("stopwords down")
from nltk.corpus import stopwords
import io
import json

from deepseekAPI import deepseek

# Create the app object
app = Flask(__name__)


# creating function for data cleaning
# from b2_preprocessing_function import CustomPreprocess
# custom = CustomPreprocess()


# Define predict function
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]
    answer = 0

    for query in query_asis:
        answer = analysis(query)
        # answer_deepseek = deepseek(query)
    sentiment = "Positive" if answer > 0.5 else "Negative"

    # 返回 JSON 数据
    return jsonify({"result": sentiment, "rate": answer})
    # return jsonify({"result": answer_deepseek, "rate": "deepseek"})


if __name__ == "__main__":
    app.run(debug=True)

