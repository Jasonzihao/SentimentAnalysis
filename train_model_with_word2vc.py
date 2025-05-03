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

from model import MAIN_Model, MAIN_PLUS_WORD2VC_Model
from preprocess import tokenize, padding_, tokenize_with_punkt
import nltk


class TextDataset(Dataset):
    def __init__(self, text_features, labels):
        super(TextDataset, self).__init__()
        # 保持数据在 CPU 上
        self.text_features = text_features
        self.labels = labels

    def __getitem__(self, index):
        inputs = self.text_features[index, :, :]
        label = self.labels[index]
        return inputs.float(), label

    def __len__(self):
        return len(self.text_features)


def get_dataloader(batch_size):
    text_features = np.load('data/word2vec_features.npy')
    labels = np.load('data/labels.npy')
    # 这里直接从 numpy 转成 tensor，数据仍在 CPU 上
    text_features = torch.from_numpy(text_features)
    labels = torch.from_numpy(labels)

    train_dataset = TextDataset(text_features=text_features[:25000], labels=labels[:25000])
    test_dataset = TextDataset(text_features=text_features[25000:50000], labels=labels[25000:50000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True,  pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True,  pin_memory=True)
    print("get dataloader successfully")
    return train_loader, test_loader



is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# dataloaders
batch_size = 128
(train_loader,valid_loader) = get_dataloader(batch_size)

no_layers = 1
embedding_dim = 32
output_dim = 1
hidden_dim = 256
model = MAIN_PLUS_WORD2VC_Model(no_layers,embedding_dim,hidden_dim)

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
            loss = criterion(outputs.reshape(32), labels.float())
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
    print(f"Test Accuracy: {test_accuracy} F1 score: {f1}")
    if test_accuracy > 0.86:
        torch.save(model.state_dict(), f'model_pt/djs_model_plusword2vc_{test_accuracy}.pt')
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
ax1.set_title("loss")
ax2.plot(acc)
ax2.set_title("accuracy")
ax1.legend()
plt.show()