from cProfile import label

import torch
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import numpy as np
import pandas as pd

from preprocess import preprocess_string
from gensim.models import Word2Vec
nltk.download("punkt_tab")

# 加载 IMDb 数据
def load_imdb_data():
    dataset = load_dataset("imdb")
    x_train = dataset['train']['text']
    x_test = dataset['test']['text']
    print(f"train length({len(x_train)}), text length({len(x_test)})")
    data = x_train + x_test
    print(f"final data length({len(data)})")
    y_train = dataset['train']['label']
    y_test = dataset['test']['label']
    label = y_train + y_test
    print(f"final labels length({len(label)})")
    return data, label


def load_word2vc():
    # 加载已训练好的 Word2Vec 模型
    word2vec_model = Word2Vec.load("word2vc_model/imdb_word2vec.model")
    # 获取词汇表大小（根据词汇表大小初始化Embedding层）
    vocab_size = len(word2vec_model.wv) + 1  # 加1是为了考虑填充token
    print(f"vocab_size: {vocab_size}")
    # 获取词向量维度
    embedding_dim = word2vec_model.vector_size
    print(f"embedding_dim: {embedding_dim}")
    # 创建嵌入矩阵，初始化为 Word2Vec 模型的词向量
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))

    # 将Word2Vec模型中的词向量填充到嵌入矩阵
    for i, word in enumerate(word2vec_model.wv.index_to_key):
        embedding_matrix[i + 1] = torch.tensor(word2vec_model.wv[word])
    # 将填充token的位置设为0
    embedding_matrix[0] = torch.zeros(embedding_dim)
    print("Load word2vc successfully!")

    return embedding_matrix



if __name__ == "__main__":
    # 加载数据并分词
    raw_texts, labels = load_imdb_data()
    tokenized_texts = [word_tokenize(text) for text in raw_texts]

    text_size = 500
    embedding_size = 32
    # 训练 Word2Vec
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_size, window=5, min_count=2, workers=4, sg=1, max_vocab_size=15000)  # sg=1 for Skip-Gram
    vocab_size = len(word2vec_model.wv) + 1  # 加1是为了考虑填充token
    print(word2vec_model.wv)
    print(f"vocab_size: {vocab_size}")
    word2vec_model.save("word2vc_model/imdb_word2vec.model")

    print("saving labels")
    np.save('data/labels.npy',labels)
    print("saving labels successfully")

    wrong_words = []
    text_features = np.zeros((50000, text_size, embedding_size))
    for j in range(len(tokenized_texts)):
        review = tokenized_texts[j]
        for i in range(len(review)):
            if i >= text_size:
                break
            try:
                text_features[j,i,:] = word2vec_model.wv[review[i]]
            except:
                continue
                # print(review[i])
                # if review[i] not in wrong_words:
                #     wrong_words.append(review[i])
                #     f = open("logs/wrong_words.txt", 'a',encoding='utf-8')
                #     f.write(review[i] + '\n')
                #     f.close()
    print("saving...")
    np.save('data/word2vec_features.npy',text_features)
    print("npy saving finished")
