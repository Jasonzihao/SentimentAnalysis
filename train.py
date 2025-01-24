import torch
import torch.nn as nn
import pandas as pd
from keras.src.ops import shape
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

from model import MAIN_Model
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
from datasets import load_dataset
dataset = load_dataset("imdb")

#create test and train split
x_train = dataset['train']['text']
y_train = dataset['train']['label']
x_test = dataset['test']['text']
y_test = dataset['test']['label']
# print(f"len of x_train:{len(x_train)}")
# print(f"len of y_train:{len(y_train)}")
# print(f"len of x_test:{len(x_test)}")
# print(f"len of y_test:{len(y_test)}")

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
# x_train,y_train,x_test,y_test,vocab = tokenize(x_train,y_train,x_test,y_test)
x_train,y_train,x_test,y_test,vocab = tokenize_with_punkt(x_train,y_train,x_test,y_test)
print("success tokenize!")

x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 32

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last=True)

#one random sample
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

no_layers = 1
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 500
output_dim = 1
hidden_dim = 500
word2vec = load_word2vc()

model = MAIN_Model(no_layers,vocab_size,embedding_dim,hidden_dim)
# model = MAIN_Model(no_layers,vocab_size,embedding_dim,hidden_dim,word2vec)

model.to(device)
print(model)
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)
sample_x = sample_x.to(device)
h = model.init_hidden(batch_size, device)
o , hh = model(sample_x,h)

max_epoch = 20
train_loss = []
test_loss = []
acc = []
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
        loss = criterion(outputs.reshape(32), labels.float())
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
            loss = criterion(outputs.reshape(32), labels.float())
            test_running_loss += loss.item()
            predicted = torch.round(outputs).squeeze()
            y_pred.extend(predicted)
            y_true.extend(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch + 1}, Loss: {test_running_loss / len(valid_loader)}")
        test_loss.append(test_running_loss / len(valid_loader))

    print(f"Test Accuracy: {correct / total}")
    acc.append(correct / total)

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_pt/djs_model_epoch_{epoch + 1}.pt')

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
ax1.set_title("loss")
ax2.plot(acc)
ax2.set_title("accuracy")
ax1.legend()
plt.show()