import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
def aaindex(sequences):
# 打开CSV文件
  with open('aaindex_feature.csv', 'r') as file:
      csv_reader = csv.reader(file)

      # 跳过标题行
      header = next(csv_reader)

      # 创建一个空字典来存储数据
      data_dict = {}

      # 遍历CSV文件中的每一行
      for row in csv_reader:
          name = row[0]  # 第一列是名称
          values = [float(value) for value in row[1:]]  # 后面的列为数值，转换为浮点数

          data_dict[name] = values

  print(data_dict)
  encodings = []
  for sequence in sequences:
    # print(sequence)
    code=[]
    for j in sequence:
      # print(j)
      code = code + data_dict[j]
      # print(len(data_dict[j]))
    encodings.append(code)
  return encodings


df=pd.read_csv('pd_main.csv',encoding='utf-8-sig',header=None,skiprows=1,names=['seq','lable'])
seq=df['seq']
# seq=["ACX","AXX"]
encode=aaindex(seq)
encode=np.array(encode)
print(seq)

aaindex=torch.tensor(encode)
print(aaindex.size())
aaindex_f=aaindex.reshape(-1,50,531)
print(aaindex_f.size())


# import numpy as npdf=pd.read_csv('pd_train.csv',encoding='utf-8-sig',header=None,skiprows=1,names=['seq','lable'])
# seq=df['seq']
# seq=np.array(seq)



def BLOSUM62(sequences):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '*': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # *
        '_': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # _
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # _
    }

    encodings = []
    for sequence in sequences:
        code=[]
        for j in sequence:
            code = code + blosum62[j]
        encodings.append(code)
    return encodings


def write_to_csv(encodings, file):
    with open(file, 'w') as f:
        for line in encodings:
            f.write(str(line[0]))
            for i in range(1, len(line)):
                f.write(',%s' % line[i])
            f.write('\n')

encode=BLOSUM62(seq)
encode=torch.tensor(encode)
encode=encode.reshape(-1,50,20)
encode.shape

import torch
print(encode.size(),aaindex_f.size())
merged_tensor=torch.cat((encode,aaindex_f),dim=-1)
print(merged_tensor.size())

import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
import pandas as pd
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

df=pd.read_csv('pdo_main_co.csv',encoding='utf-8-sig',header=None,skiprows=1,names=['seq','lable'])
# df_t=pd.read_csv('pro_test.csv',encoding='utf-8-sig',header=None,skiprows=1,names=['seq','lable'])
# df = pd.concat([df, df_t])

seq=df['seq']
seq = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq]
# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
# ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, pad_to_max_length=True)
ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, max_length=50,pad_to_max_length=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)
# seq_t=df_t['seq']
# seq_t = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_t]
# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
# ids_t = tokenizer.batch_encode_plus(seq_t, add_special_tokens=True, pad_to_max_length=True)
# # input_ids_t = torch.tensor(ids_t['input_ids']).to(device)
# attention_mask_t = torch.tensor(ids_t['attention_mask']).to(device)
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
embedding = embedding.cpu().numpy()
# features = []
features=embedding
# for seq_num in range(len(embedding)):
#     seq_len = (attention_mask[seq_num] == 1).sum()
#     seq_emd = embedding[seq_num][1:seq_len-1]
#     features.append(seq_emd)

# with torch.no_grad():
#     embedding_t = model(input_ids=input_ids_t,attention_mask=attention_mask_t)[0]
# embedding_t = embedding_t.cpu().numpy()
# features_t = []
# for seq_num in range(len(embedding_t)):
#     seq_len = (attention_mask_t[seq_num] == 1).sum()
#     seq_emd = embedding_t[seq_num][1:seq_len-1]
#     features_t.append(seq_emd)



print(torch.tensor(features).size())

merged_tensor=torch.cat((merged_tensor,torch.tensor(features)),dim=-1)
print(merged_tensor.size())


reduced_data=torch.tensor(merged_tensor)
y=df['lable']
y=torch.tensor(y)
batch_size=16
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
X,X_test,y,y_test=train_test_split(reduced_data,y,test_size=0.2,random_state=414)


from sklearn.decomposition import PCA
n_components=256
model_pca = PCA(n_components=256)
X= X.reshape((X.shape[0],X.shape[1],X.shape[2]))
X = model_pca.fit_transform(X.reshape(-1, X.shape[-1]))
print(f"降维后的特征维度数：{model_pca.n_components_}")
X=X.reshape(-1,50,n_components)
X_test=model_pca.transform(X_test.reshape(-1,X_test.shape[-1]))
X_test=X_test.reshape(-1,50,n_components)
print(X.shape)

from joblib import dump

# Save the model
dump(model_pca, 'pca_model_set2.joblib')

# reduced_data=torch.tensor(reduced_data)
# y=df['lable']
# y=torch.tensor(y)
batch_size=16
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# X,X_test,y,y_test=train_test_split(reduced_data,y,test_size=0.2,random_state=418)
# dataset = TensorDataset(X.double(), y)
# test_dataset=TensorDataset(X_test.double(),y_test)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
X=torch.tensor(X)
X_test=torch.tensor(X_test)
dataset = TensorDataset(X.double(), y)
test_dataset=TensorDataset(X_test.double(),y_test)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
USE_CUDA = True

import model

from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
def cal_acc(y_true,X_test,net):
  test_dataset = TensorDataset(X_test,torch.tensor(y_true))
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  with torch.no_grad():
    all_pred = []
    all_score = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        output, reconstructions, masked = net(batch_x.to("cuda"))
        #score = logit.squeeze(1).cpu().detach().numpy().tolist()
        #all_score += score
        #pred_labels = np.argmax(masked.data.cpu().numpy(), 1)
        pred  = np.argmax(masked.cpu().detach().numpy(),axis=1).tolist()
        # print(pred)
        all_pred += pred
    tn, fp, fn, tp = confusion_matrix(y_test, all_pred).ravel()
    #fpr, tpr, _ = roc_curve(test_labels, all_pred)
    #aucroc = auc(fpr, tpr)
    perftab = {"CM": confusion_matrix(y_test, all_pred),
            'ACC': (tp + tn) / (tp + fp + fn + tn),
            'SEN': tp / (tp + fn),
            'PREC': tp / (tp + fp),
            "SPEC": tn / (tn + fp),
            "MCC": matthews_corrcoef(y_test, all_pred),
            "F1": f1_score(y_test, all_pred)
    }
    acc=perftab['ACC']
    recall=perftab['SEN']
    perc=perftab['PREC']
    return acc,recall,perc,perftab

  # from torch.optim import SGD
  capsule_net = CapsNet()
  if USE_CUDA:
      capsule_net = capsule_net.cuda()
  optimizer = Adam(capsule_net.parameters(), lr=0.001, betas=(0.9, 0.999))
  # optimizer = SGD(capsule_net.parameters(),lr=0.01)

  import numpy as np

  batch_size = 16
  n_epochs = 250
  best_acc = 0
  for epoch in range(n_epochs):
      capsule_net.train()
      train_loss = 0
      correct_train = 0
      TP_train, FN_train, FP_train = 0, 0, 0

      for batch_id, (data, target) in enumerate(train_loader):
          target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
          data, target = Variable(data), Variable(target)

          if USE_CUDA:
              data, target = data.cuda(), target.cuda()

          optimizer.zero_grad()
          output, reconstructions, masked = capsule_net(data)
          loss = capsule_net.loss(data, output, target, reconstructions)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()

          # Calculate the number of correct predictions
          pred_labels = np.argmax(masked.data.cpu().numpy(), 1)
          true_labels = np.argmax(target.data.cpu().numpy(), 1)
          correct_train += np.sum(pred_labels == true_labels)

          # Calculate TP, FN, FP for recall calculation
          TP_train += np.sum((pred_labels == 1) & (true_labels == 1))
          FN_train += np.sum((pred_labels == 0) & (true_labels == 1))
          FP_train += np.sum((pred_labels == 1) & (true_labels == 0))

      # Calculate average accuracy, loss, and recall for the epoch
      avg_train_accuracy = correct_train / len(train_loader.dataset)
      avg_train_loss = train_loss / len(train_loader)
      recall_train = TP_train / (TP_train + FN_train)
      precision_train = TP_train / (TP_train + FP_train)

      print(
          f"Epoch {epoch + 1}/{n_epochs}, Train Accuracy: {avg_train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}, Train Recall: {recall_train:.4f},Train Precision :{precision_train:.4f}")

      capsule_net.eval()
      with torch.inference_mode():
          avg_test_accuracy, recall_test, precision_test, tab = cal_acc(y_test, X_test, capsule_net)
          print(
              f"Test Accuracy: {avg_test_accuracy:.4f}, Test Recall: {recall_test:.4f},Test Precision:{precision_test:.4f}")
          # test_loss = 0
          # correct_test = 0
          # TP_test, FN_test, FP_test = 0, 0, 0

          # for batch_id, (data, target) in enumerate(test_loader):
          #     target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
          #     data, target = Variable(data), Variable(target)

          #     if USE_CUDA:
          #         data, target = data.cuda(), target.cuda()

          #     output, reconstructions, masked = capsule_net(data.double())
          #     test_loss += capsule_net.loss(data, output, target, reconstructions).item()

          #     # Calculate the number of correct predictions
          #     pred_labels = np.argmax(masked.data.cpu().numpy(), 1)
          #     true_labels = np.argmax(target.data.cpu().numpy(), 1)
          #     correct_test += np.sum(pred_labels == true_labels)

          #     # Calculate TP, FN, FP for recall calculation
          #     TP_test += np.sum((pred_labels == 1) & (true_labels == 1))
          #     FN_test += np.sum((pred_labels == 0) & (true_labels == 1))
          #     FP_test += np.sum((pred_labels == 1) & (true_labels == 0))

          # # Calculate average accuracy, loss, and recall for the epoch
          # avg_test_accuracy = correct_test / len(test_loader.dataset)
          # avg_test_loss = test_loss / len(test_loader)
          # recall_test = TP_test / (TP_test + FN_test)
          # precision_test=TP_test/(TP_test+FP_test)

          # print(f"Test Accuracy: {avg_test_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Recall: {recall_test:.4f},Test Precision:{precision_test:.4f}")
          if best_acc < avg_test_accuracy:
              best_acc = avg_test_accuracy
              torch.save(capsule_net, 'capsule_net.pth')