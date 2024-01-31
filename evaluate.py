import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
import train

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

import numpy as np
import pandas as pd
batch_size = 64
res = 64
fig = pd.read_csv("data/stage1/Train.txt")

str_list=[]
for i in fig['figure']:
  temp=np.array(i.split(" "),dtype=np.float32).reshape(res,res)
  str_list.append(temp)
y=fig['label']

fig_test = pd.read_csv("data/stage1/Test.txt")
y_test=fig_test['label']
str_list_test=[]
for i in fig_test['figure']:
  temp=np.array(i.split(" "),dtype=np.float32).reshape(res,res)
  str_list_test.append(temp)

X=torch.tensor(str_list)
X_test=torch.tensor(str_list_test)
X=X.unsqueeze(1)
X_test=X_test.unsqueeze(1)


from torch.utils.data import DataLoader, TensorDataset
dataset=TensorDataset(X,torch.tensor(y))
test_dataset=TensorDataset(X_test,torch.tensor(y_test))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_cap=torch.load('models/capsule_net_stage1.pth')#plase change the path of the model
dataset=TensorDataset(X,torch.tensor(y))
test_dataset = TensorDataset(X_test,torch.tensor(y_test))
train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
with torch.no_grad():
    all_pred = []
    all_score = []
    fe = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        output, reconstructions, masked = model_cap(batch_x.to("cuda"))
        pred  = np.argmax(masked.cpu().detach().numpy(),axis=1).tolist()
        print(pred)
        classes = torch.sqrt((output ** 2).sum(2))
        classes = F.softmax(classes,dim=1)
        all_pred += pred
        fe += output
    tn, fp, fn, tp = confusion_matrix(y_test, all_pred).ravel()
    perftab = {"CM": confusion_matrix(y_test, all_pred),
            'ACC': (tp + tn) / (tp + fp + fn + tn),
            'SEN': tp / (tp + fn),
            'PREC': tp / (tp + fp),
            "SPEC": tn / (tn + fp),
            "MCC": matthews_corrcoef(y_test, all_pred),
            "F1": f1_score(y_test, all_pred)
    }