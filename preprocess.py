import os

import nltk
import torch
import torch.nn as nn
import pandas as pd
from keras.src.backend import shape
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn import preprocessing
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import json
from nltk.tokenize import word_tokenize
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import gensim
import gensim.downloader as api





def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tokenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:15000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                    if preprocess_string(word) in onehot_dict.keys()])

    final_list_train = padding_(final_list_train, 500)
    final_list_test = padding_(final_list_test, 500)
    print(len(final_list_train))
    print(len(final_list_test))
    # 保存 onehot_dict 到本地
    with open('onehot_dict.json', 'w') as f:
        json.dump(onehot_dict, f)

    return np.array(final_list_train), np.array(y_train),np.array(final_list_test), np.array(y_val),onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def tokenize_with_punkt(x_train, y_train, x_val, y_val):
    word_list = []
    stop_words = set(stopwords.words('english'))
    if os.path.exists('onehot_dict_punk.json'):
        with open('onehot_dict.json', 'r') as f:
            onehot_dict = json.load(f)
    else:
        # 使用 Punkt 分词器处理训练数据
        for sent in x_train:
            words = word_tokenize(sent.lower())  # 使用 Punkt 分词器
            for word in words:
                word = preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)

        # 构建词汇表
        corpus = Counter(word_list)
        corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:15000]
        onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}
        # 保存词汇表到本地
        with open('onehot_dict_punk.json', 'w') as f:
            json.dump(onehot_dict, f)

    # 对训练集和验证集进行分词和编码
    final_list_train, final_list_test = [], []
    for sent in x_train:
        words = word_tokenize(sent.lower())  # 使用 Punkt 分词器
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in words
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        words = word_tokenize(sent.lower())  # 使用 Punkt 分词器
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in words
                                if preprocess_string(word) in onehot_dict.keys()])

    # 对分词后的结果进行填充
    final_list_train = padding_(final_list_train, 500)
    final_list_test = padding_(final_list_test, 500)
    print(final_list_train[0])
    print(final_list_test[0])
    print(len(final_list_train))
    print(len(final_list_test))



    return np.array(final_list_train), np.array(y_train), np.array(final_list_test), np.array(y_val), onehot_dict





