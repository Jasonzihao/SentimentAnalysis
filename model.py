import torch
import torch.nn as nn
import pandas as pd
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
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import gensim
import gensim.downloader as api



# class MAIN_Model(nn.Module):
#
#     def __init__(self,no_layers,vocab_size,embedding_dim,hidden_dim):
#         super(MAIN_Model, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5)
#
#         self.conv_bn = nn.BatchNorm1d(64)
#
#         self.dropout = nn.Dropout(p=0.5)
#
#         self.global_pool = nn.AdaptiveMaxPool1d(1)
#
#         self.pool1 = nn.MaxPool1d(kernel_size=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=5)
#         self.pool3 = nn.MaxPool1d(kernel_size=7)
#         self.lstm = nn.LSTM(input_size=334,hidden_size=hidden_dim,num_layers=no_layers,bidirectional = True, batch_first=True)
#
#         self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
#
#         self.fc = nn.Linear(hidden_dim * 2, 100)
#
#         self.fc_bn = nn.BatchNorm1d(100)
#
#         self.fc2 = nn.Linear(100, 1)
#         self.embedding_dim = embedding_dim
#         self.no_layers = no_layers
#         self.vocab_size = vocab_size
#         self.hidden_dim = hidden_dim
#
#
#     def attention_net(self, lstm_output, final_state):
#         #lstm_output = lstm_output.permute(1, 0, 2)
#         hidden = final_state.squeeze(0)
#         attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
#         soft_attn_weights = F.softmax(attn_weights, dim=1)
#         new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
#                                      soft_attn_weights.unsqueeze(2)).squeeze(2)
#
#         return new_hidden_state
#
#     def attention(self, lstm_output, final_state):
#       #  lstm_output = lstm_output.permute(1, 0, 2)
#
#         merged_state = torch.cat([s for s in final_state], 1)
#         # merged_state = merged_state.squeeze(0).unsqueeze(2)
#
#         # print("Before squeeze: ", merged_state.shape)
#         # merged_state = merged_state.squeeze(0)
#         # print("After squeeze: ", merged_state.shape)
#         merged_state = merged_state.unsqueeze(2)
#         # print("After squeeze: ", merged_state.shape)
#
#         weights = torch.bmm(lstm_output, merged_state)
#         weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
#         return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)
#
#     def forward(self, x, hidden):
#         x = self.embedding(x)
#         x = x.reshape(len(x), self.embedding_dim, 500)
#         x = self.conv1(x)
#
#         x = self.conv_bn(x)
#
#         x = nn.functional.tanh(x)
#         x1 = self.pool1(x)
#         x2 = self.pool2(x)
#         x3 = self.pool3(x)
#         x = torch.cat((x1,x2,x3),2)
#         x = self.dropout(x)
#         output, (hidden, cell) = self.lstm(x, hidden)
#
#         output = self.lstm_norm(output)
#
#         attn_output = self.attention(output, hidden)
#         out = self.fc(attn_output.squeeze(0))
#
#         out = self.fc_bn(out)
#
#         #x = self.dropout(x)
#         out = self.fc2(out)
#         out = nn.functional.sigmoid(out)
#         return out, hidden
#
#
#     def init_hidden(self, batch_size, device):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
#         # initialized to zero, for hidden state and cell state of LSTM
#         h0 = torch.zeros((self.no_layers *2,batch_size,self.hidden_dim)).to(device)
#         c0 = torch.zeros((self.no_layers *2,batch_size,self.hidden_dim)).to(device)
#         hidden = (h0,c0)
#         return hidden



class MAIN_Model(nn.Module):

    def __init__(self, no_layers, vocab_size, embedding_dim, hidden_dim):
        super(MAIN_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1, padding=2)  # 保持输入输出维度一致
        self.conv_bn1 = nn.BatchNorm1d(64)  # 对第1层卷积进行BatchNorm

        # 将池化操作替换为卷积操作
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn2 = nn.BatchNorm1d(128)  # 对第2层卷积进行BatchNorm

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn3 = nn.BatchNorm1d(256)  # 对第3层卷积进行BatchNorm

        self.dropout = nn.Dropout(p=0.3)

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=no_layers, bidirectional=True, batch_first=True)

        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(hidden_dim * 2, 100)

        self.fc_bn = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 1)

        self.embedding_dim = embedding_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.unsqueeze(2)

        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = x.reshape(len(x), self.embedding_dim, 500)

        # 第一层卷积 + BatchNorm
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = nn.functional.relu(x)

        # 第二层卷积 + BatchNorm
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = nn.functional.relu(x)

        # 第三层卷积 + BatchNorm
        x = self.conv3(x)
        x = self.conv_bn3(x)
        x = nn.functional.relu(x)

        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x.transpose(1, 2), hidden)  # 需要调整维度为 (batch_size, seq_len, input_size)


        attn_output = self.attention(output, hidden)

        attn_output = self.fc_dropout(attn_output)

        out = self.fc(attn_output.squeeze(0))
        out = self.fc2(out)
        out = nn.functional.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        h0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


