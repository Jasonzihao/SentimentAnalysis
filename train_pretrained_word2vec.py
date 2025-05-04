import torch
import torch.nn as nn
import pandas as pd
from keras.src.ops import shape
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn import preprocessing
from torchsummary import summary
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors

from model import MAIN_Model, MAIN_PLUS_WORD2VC_Model, MAIN_Model_pretrained_embedding
from preprocess import tokenize, padding_, tokenize_with_punkt
from train_word2vc import load_word2vc

is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#load dataset
from datasets import load_dataset, load_from_disk

# 下载并加载预训练Word2Vec模型（这里使用Google News 300d，首次运行会下载约3.4GB文件）
# 也可以使用更小的模型如'glove-wiki-gigaword-300'或'word2vec-ruscorpora-300'
# 如果网络问题，可以手动下载后指定路径
pretrained_model_name = 'word2vec-google-news-300'
wv = api.load(pretrained_model_name)
# 构建预训练嵌入矩阵
embedding_dim = 300  # 匹配预训练模型维度
vocab_size = 30000
pretrained_embedding = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
# 填充已知词：
# - padding_idx=0 保持全0
# - 1到len(wv.vocab)为正常词汇
# - len(wv.vocab)+1为OOV词（索引从0开始，需注意索引对应）
print("get word2vec model...")
# 遍历key_to_index获取单词和索引（替代原vocab）
for word, idx_in_wv in wv.key_to_index.items():
    # 原逻辑：索引0为padding，实际单词从索引1开始填充
    # idx_in_wv：预训练模型中的原始索引（从0开始）
    # 目标位置：1 <= target_idx < vocab_size-1（留出最后一位给OOV）
    target_idx = idx_in_wv + 1  # 跳过padding的0索引
    if target_idx < vocab_size - 1:  # 避免超出自定义词表大小
        pretrained_embedding[target_idx] = wv[word]
# OOV词处理：索引vocab_size-1（最后一位）
pretrained_embedding[-1] = np.random.normal(scale=0.01, size=embedding_dim)
print("get word2vec model successfully")

dataset = load_dataset("imdb")
# dataset = load_from_disk("imdb")
#create test and train split
x_train = dataset['train']['text']
y_train = dataset['train']['label']
x_test = dataset['test']['text']
y_test = dataset['test']['label']
# x_train = dataset['train']['text'] + x_test[:15000]
# y_train = dataset['train']['label'] + y_test[:15000]
print(f"len of x_train:{len(x_train)}")
print(f"len of y_train:{len(y_train)}")
print(f"len of x_test:{len(x_test)}")
print(f"len of y_test:{len(y_test)}")

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
# x_train,y_train,x_test,y_test,vocab = tokenize(x_train,y_train,x_test,y_test)
x_train_pad,y_train,x_test_pad,y_test,_ = tokenize_with_punkt(x_train,y_train,x_test,y_test)
print("success tokenize!")

# x_train_pad = padding_(x_train,500)
# x_test_pad = padding_(x_test,500)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 32

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)

#one random sample
# dataiter = iter(train_loader)
# sample_x, sample_y = next(dataiter)

no_layers = 1
print(f"vocab_size:{vocab_size}")

embedding_dim = 300
output_dim = 1
hidden_dim = 500
model = MAIN_Model_pretrained_embedding(no_layers,embedding_dim,hidden_dim,pretrained_embedding)

model.to(device)
print(model)
# dataiter = iter(train_loader)
# sample_x, sample_y = next(dataiter)
# sample_x = sample_x.to(device)
# h = model.init_hidden(batch_size, device)
# o , hh = model(sample_x,h)

max_epoch = 20
train_loss = []
test_loss = []
acc = []
f1_scores = []
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# Train the model
for epoch in range(max_epoch):
    a = 0
    running_loss = 0.0
    h = model.init_hidden(batch_size, device)
    for i, data in tqdm(enumerate(train_loader, 0)):
        a = a+1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #h = tuple([each.data for each in h])
        h = model.init_hidden(batch_size, device)
        optimizer.zero_grad()
        outputs,h = model(inputs,h)
        #print(outputs.shape)
        loss = criterion(outputs.reshape(batch_size), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    train_loss.append(running_loss / len(train_loader))

# Evaluate the model
    correct = 0
    total = 0
    test_running_loss = 0.0
    y_pred = []
    y_true = []
    val_h = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for data in valid_loader:
            #val_h = tuple([each.data for each in val_h])
            val_h = model.init_hidden(batch_size, device)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,val_h = model(inputs,val_h)
            loss = criterion(outputs.reshape(batch_size), labels.float())
            test_running_loss += loss.item()
            predicted = torch.round(outputs).squeeze()
            true_labels = labels
            y_pred.extend(predicted.cpu().tolist())
            y_true.extend(true_labels.cpu().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
        print(f"Epoch {epoch + 1}, Loss: {test_running_loss / len(valid_loader)}")
        test_loss.append(test_running_loss / len(valid_loader))
    test_accuracy = (correct / total)
    acc.append(test_accuracy)
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)
    print(f"Test Accuracy: {test_accuracy}, F1_score: {f1}")
    if test_accuracy > 0.862:
        torch.save(model.state_dict(), f'model_pt/djs_model_epoch_{test_accuracy}_em{embedding_dim}_hi{hidden_dim}_ba{batch_size}.pt')
    # if (epoch + 1) % 5 == 0:
    #     torch.save(model.state_dict(), f'model_pt/djs_model_epoch_{epoch + 1}.pt')

    classes = ('NEG', 'POS')

# Build confusion matrix
#yt = y_true.cpu()
cf_matrix = confusion_matrix(torch.tensor(y_true).cpu(),torch.tensor(y_pred).cpu() )
df_cm = pd.DataFrame(cf_matrix , index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt='g')

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6))
fig.tight_layout(pad=5.0)
ax1.plot(train_loss, label="train loss")
ax1.plot(test_loss,label="test loss")
ax1.plot(f1_scores, label="F1 score")
ax1.set_title("loss")
ax2.plot(acc)
ax2.set_title("accuracy")
ax1.legend()
plt.show()