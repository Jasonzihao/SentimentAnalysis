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




class MAIN_Model(nn.Module):

    def __init__(self, no_layers, vocab_size, embedding_dim, hidden_dim):
        super(MAIN_Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1, padding=2)  # 保持输入输出维度一致
        self.conv_bn1 = nn.BatchNorm1d(64)  # 对第1层卷积进行BatchNorm

        # 将池化操作替换为卷积操作
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn2 = nn.BatchNorm1d(128)  # 对第2层卷积进行BatchNorm

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn3 = nn.BatchNorm1d(256)  # 对第3层卷积进行BatchNorm

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=no_layers, bidirectional=True, batch_first=True)
        # 在 LSTM 输出后添加归一化层
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)

        self.fc_dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(hidden_dim * 2, 100)

        self.fc_bn = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 1)

        self.embedding_dim = embedding_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim


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
        # 对 LSTM 输出应用归一化
        output = self.lstm_bn(output.transpose(1, 2)).transpose(1, 2)

        # 使用 Attention 机制
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


class MAIN_PLUS_WORD2VC_Model(nn.Module):

    def __init__(self, no_layers, embedding_dim, hidden_dim):
        super(MAIN_PLUS_WORD2VC_Model, self).__init__()

        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1, padding=2)  # 保持输入输出维度一致
        self.conv_bn1 = nn.BatchNorm1d(64)  # 对第1层卷积进行BatchNorm

        # 将池化操作替换为卷积操作
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn2 = nn.BatchNorm1d(128)  # 对第2层卷积进行BatchNorm

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 输出通道数增大
        self.conv_bn3 = nn.BatchNorm1d(256)  # 对第3层卷积进行BatchNorm

        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=no_layers, bidirectional=True, batch_first=True)
        # 在 LSTM 输出后添加归一化层
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc_dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(hidden_dim * 2, 100)
        self.fc2 = nn.Linear(100, 1)

        self.embedding_dim = embedding_dim
        self.no_layers = no_layers
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
        # 对 LSTM 输出应用归一化
        output = self.lstm_bn(output.transpose(1, 2)).transpose(1, 2)

        # 使用 Attention 机制
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

class MAIN_Model_pretrained_embedding(nn.Module):

    def __init__(self, no_layers, embedding_dim, hidden_dim, pretrained_weights):
        super(MAIN_Model_pretrained_embedding, self).__init__()

        # 使用预训练权重创建嵌入层
        self.embedding = nn.Embedding(len(pretrained_weights), embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_weights), requires_grad=True)  # 冻结权重
        # 第一层卷积块：卷积 + BN + ReLU + 最大池化
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1, padding=2)  # 保持输入输出长度一致
        self.conv_bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化后长度减半（如500→250）

        # 第二层卷积块：卷积 + BN + ReLU + 最大池化
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 保持长度不变（250→250）
        self.conv_bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化后长度250→125

        # 第三层卷积块：卷积 + BN + ReLU + 最大池化（可选：最后一层池化可保留长度）
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 保持长度不变（125→125）
        self.conv_bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)  # 池化后125→63（向上取整避免小数）

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=no_layers, bidirectional=True, batch_first=True)
        # 在 LSTM 输出后添加归一化层
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)

        self.fc_dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(hidden_dim * 2, 100)

        self.fc_bn = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 1)

        self.embedding_dim = embedding_dim
        self.no_layers = no_layers
        self.hidden_dim = hidden_dim


    def attention(self, lstm_output, final_state):
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.unsqueeze(2)

        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x, hidden):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # 转为卷积输入格式：(batch, channel, seq_len)

        # 第一层卷积块
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # 池化后seq_len减半（如500→250）

        # 第二层卷积块
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # 池化后seq_len再减半（250→125）

        # 第三层卷积块
        x = self.conv3(x)
        x = self.conv_bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # 池化后seq_len→63（ceil_mode=True处理奇数长度）

        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 转为LSTM输入格式：(batch, seq_len, channel)

        output, (hidden, cell) = self.lstm(x, hidden)
        output = self.lstm_bn(output.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化维度调整

        attn_output = self.attention(output, hidden)
        attn_output = self.fc_dropout(attn_output)

        # 注意：squeeze(0)可能导致维度错误，改为直接线性层输入
        out = self.fc(attn_output)
        out = self.fc_bn(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        h0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


class MAIN_Model_maxPooling(nn.Module):

    def __init__(self, no_layers, vocab_size, embedding_dim, hidden_dim):
        super(MAIN_Model_maxPooling, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 第一层卷积块：卷积 + BN + ReLU + 最大池化
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=5, stride=1, padding=2)  # 保持输入输出长度一致
        self.conv_bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化后长度减半（如500→250）

        # 第二层卷积块：卷积 + BN + ReLU + 最大池化
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 保持长度不变（250→250）
        self.conv_bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化后长度250→125

        # 第三层卷积块：卷积 + BN + ReLU + 最大池化（可选：最后一层池化可保留长度）
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 保持长度不变（125→125）
        self.conv_bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)  # 池化后125→63（向上取整避免小数）

        self.dropout = nn.Dropout(p=0.4)

        # LSTM输入维度：池化后序列长度为63（假设初始500→250→125→63）
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=no_layers,
            bidirectional=True,
            batch_first=True
        )
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)

        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 100)
        self.fc_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 1)

        self.embedding_dim = embedding_dim
        self.no_layers = no_layers
        self.hidden_dim = hidden_dim

    def attention(self, lstm_output, final_state):
        # 注意力机制不变
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x, hidden):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # 转为卷积输入格式：(batch, channel, seq_len)

        # 第一层卷积块
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # 池化后seq_len减半（如500→250）

        # 第二层卷积块
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # 池化后seq_len再减半（250→125）

        # 第三层卷积块
        x = self.conv3(x)
        x = self.conv_bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # 池化后seq_len→63（ceil_mode=True处理奇数长度）

        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 转为LSTM输入格式：(batch, seq_len, channel)

        output, (hidden, cell) = self.lstm(x, hidden)
        output = self.lstm_bn(output.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化维度调整

        attn_output = self.attention(output, hidden)
        attn_output = self.fc_dropout(attn_output)

        # 注意：squeeze(0)可能导致维度错误，改为直接线性层输入
        out = self.fc(attn_output)
        out = self.fc_bn(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out, hidden

    # 初始化隐藏层不变
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2, batch_size, self.hidden_dim)).to(device)
        return (h0, c0)

