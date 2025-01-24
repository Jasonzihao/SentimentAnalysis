import torch
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
nltk.download("punkt_tab")

# 加载 IMDb 数据
def load_imdb_data():
    dataset = load_dataset("imdb")
    x_train = dataset['train']['text']
    return x_train


def load_word2vc():
    # 加载已训练好的 Word2Vec 模型
    word2vec_model = Word2Vec.load("word2vc_model/imdb_word2vec.model")
    # 获取词汇表大小（根据词汇表大小初始化Embedding层）
    vocab_size = len(word2vec_model.wv) + 1  # 加1是为了考虑填充token
    print(f"vocab_size{vocab_size}")
    # 获取词向量维度
    embedding_dim = word2vec_model.vector_size
    # 创建嵌入矩阵，初始化为 Word2Vec 模型的词向量
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))

    # 将Word2Vec模型中的词向量填充到嵌入矩阵
    for i, word in enumerate(word2vec_model.wv.index_to_key):
        embedding_matrix[i + 1] = torch.tensor(word2vec_model.wv[word])
    # 将填充token的位置设为0
    embedding_matrix[0] = torch.zeros(embedding_dim)
    print("Load word2vc successfully!")

    return embedding_matrix


# 加载数据并分词
raw_texts = load_imdb_data()
tokenized_texts = [word_tokenize(text) for text in raw_texts]
from gensim.models import Word2Vec
# 训练 Word2Vec
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=300, window=5, min_count=2, workers=4, sg=1)  # sg=1 for Skip-Gram
word2vec_model.save("word2vc_model/imdb_word2vec.model")
