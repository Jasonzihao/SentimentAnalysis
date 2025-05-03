import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.src.ops import shape
from torch.utils.data import TensorDataset, DataLoader
import nltk
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from model import MAIN_Model
from preprocess import preprocess_string, padding_


def load_model(model_path, no_layers, vocab_size, embedding_dim, hidden_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAIN_Model(no_layers, vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


def tokenize_analysis(sentence, onehot_dict):
    tokenized_sentence = [onehot_dict[preprocess_string(word)] for word in sentence.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()]
    # print(tokenized_sentence)

    tokenized_sentence = padding_([tokenized_sentence], 500)
    # 将 numpy 数组转换为 PyTorch 张量
    tokenized_sentence = torch.from_numpy(tokenized_sentence).long()
    # print(tokenized_sentence.shape)
    return tokenized_sentence



def analysis(query):
    print(f"query is {query}")
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    # 加载预训练的词汇表，这里假设你在训练时已经保存了词汇表
    with open('onehot_dict_punk.json', 'r') as f:
        vocab = json.load(f)
    # 加载模型
    model_path = 'model_pt/djs_model_epoch_0.869598271446863_em500_hi500_ba32.pt'  # 请将此路径修改为你实际保存的模型文件路径
    no_layers = 1
    vocab_size = len(vocab) + 1
    embedding_dim = 500
    hidden_dim = 500
    model = load_model(model_path, no_layers, vocab_size, embedding_dim, hidden_dim)

    # 输入一句话
    # input_text22 = query
    # input_text = "Interstellar is a masterpiece that transcends the boundaries of traditional science - fiction films. Christopher Nolan's direction is nothing short of brilliant, crafting a visually stunning and intellectually stimulating journey through space and time. The special effects are mind - blowing, from the breathtaking view of black holes to the emotional portrayal of the vastness of the universe. Matthew McConaughey and Anne Hathaway deliver powerful performances, making the audience deeply empathize with their characters' struggles and sacrifices. The film also delves into profound themes like love, family, and the nature of humanity, leaving a lasting impression. It's a must - watch for anyone who appreciates great cinema."
    # input_text2 = "[Movie Title] offers a blend of highs and lows that make it a memorable yet flawed watch.On the plus side, the visuals are stunning. The movie opens with breathtaking landscapes and fantastical settings that draw you in immediately. The action scenes, especially the battles, are choreographed and filmed with such intensity that you can't help but be on the edge of your seat. The acting is also solid. The lead actors bring their characters to life with conviction, making you care about their fates.However, the plot is a major letdown. It's full of holes and jumps around in a way that's hard to follow. The pacing is off too; some parts drag on while others seem rushed. Also, the movie fails to develop its themes fully. It tries to cover too much ground but ends up skimming the surface.In conclusion, [Movie Title] has the potential to be great but stumbles due to its weak plot and pacing. It's a movie that shows flashes of brilliance but ultimately leaves you wanting more."
    input_text = query
    input_tensor = tokenize_analysis(input_text, vocab)
    # 添加批量维度 (batch_size=1)
    input_tensor = input_tensor.to(device)
    # print(input_tensor.shape)

    # 初始化隐藏状态 (batch_size=1)
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    # 进行预测
    with torch.no_grad():
        output, hidden = model(input_tensor, hidden)
        print(f"output:{output[0]}")
        prediction = torch.round(output).item()
        print(f"Prediction: {'POS' if prediction == 1 else 'NEG'}")
    answer = round(float(output[0].cpu().item()), 5)
    return answer

if __name__ == "__main__":
    analysis("222")