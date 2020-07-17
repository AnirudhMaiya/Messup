import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import transforms
from PreActResNet import PreActResNet,PreActBlock

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

NUM_CLASSES = 10
batch_is = 125
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_is,
                                          shuffle=True, num_workers=1,pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_is,
                                         shuffle=False, num_workers=1,pin_memory=True)

def accuracy(y_hat,y_true,lam,is_train):
  if is_train:
    labels1,labels2 = y_true[0],y_true[1]
    labels1 = F.softmax(labels1,dim = 1)
    labels2 = F.softmax(labels2,dim = 1)
    _, labels1 = torch.max(labels1, 1)
    _, labels2 = torch.max(labels2, 1)
    y_hat = F.softmax(y_hat,dim = 1)
    _, predicted = torch.max(y_hat, 1)
    a_is = lam * predicted.eq(labels1).cpu().sum()
    b_is = (1 - lam) * predicted.eq(labels2).cpu().sum()
    total_correct = a_is + b_is
    return total_correct
  else:
     y_hat = F.softmax(y_hat,dim = 1)
     _, predicted = torch.max(y_hat, 1)
     total_correct = (predicted.reshape(-1,1) == y_true.reshape(-1,1)).sum().item()
     return total_correct

def messup(prev_values,cur,x_cur,y_cur,x_prev,alpha_is,iter_is,is_first = False):
  gas = torch.eye(NUM_CLASSES).to(device)
  if is_first:
    cur = alpha_is * gas[y_cur]
    prev_values.append(cur)
    return alpha_is * x_cur,prev_values,cur
  else:
    prev_values_cache = [((1-alpha_is)**(idx+1))*elem for idx,elem in enumerate(prev_values)]
    prev_values.insert(0,alpha_is*gas[y_cur])
    prev_values_sum = sum(prev_values_cache)
    return1,return2 = alpha_is*gas[y_cur],((1-alpha_is)**iter_is)*cur + prev_values_sum
    cur = alpha_is*gas[y_cur]+((1-alpha_is)**iter_is)*cur + prev_values_sum
    mixed_x = alpha_is * x_cur + (1-alpha_is)* x_prev
    return mixed_x,return1,return2,prev_values,cur

def cce_loss(y_pred,y_true):
  y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
  return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def mixup_criterion(y_a, y_b):
    return lambda cce_loss, pred: cce_loss(pred, y_a) + cce_loss(pred, y_b)

def train(model,epochs,loader,alpha_is):
  model.train()
  correct = 0
  loss_list = []
  keep_track_inp = {}
  keep_track_lab = {}
  for idx,(i,j) in enumerate(loader):
    inputs,labels = i.to(device),j.to(device)
    if idx % RESETCYCLE == 0: 
      keep_track_inp = {}
      keep_track_lab = {}
      count = 0
      prev_values = []
      mini_x,prev_values,cur= messup(prev_values,_,inputs,labels,_,alpha_is,idx,True)
      opt.zero_grad()
      outputs = model(mini_x)
      loss_is = cce_loss(F.softmax(outputs),prev_values[0])
      loss_is.backward()
      opt.step()
      keep_track_inp[count] = torch.clone(mini_x).detach()
      count = count + 1
      correct = correct + accuracy(outputs,labels,_,False)
    else:
      mini_x,mini_y_a,mini_y_b,prev_values,cur = messup(prev_values,cur,inputs,labels,keep_track_inp[count-1],
                                                                alpha_is,idx)
      opt.zero_grad()
      outputs = model(mini_x)
      loss_func = mixup_criterion(mini_y_a, mini_y_b)
      loss_is = loss_func(cce_loss, F.softmax(outputs))
      loss_is.backward()
      opt.step()
      keep_track_inp[count] = torch.clone(mini_x).detach()
      count = count + 1
      tot_labels = [mini_y_a,mini_y_b]
      correct = correct + accuracy(outputs,tot_labels,alpha_is,True)
    loss_list.append(loss_is.item())
  mean_loss = sum(loss_list)/len(loss_list)
  mean_acc = (correct/len(loader.dataset))*100
  print("[%d/%d] Training Accuracy : %f and Train Loss : %f "%(epochs,total_epochs,mean_acc,mean_loss))
  return mean_loss,mean_acc

def test(model,epochs,loader,is_train = True):
  model.eval()
  correct = 0
  loss_list = []
  with torch.no_grad():
    for i,j in loader:
      inputs,labels = i.to(device),j.to(device)
      outputs = model(inputs)
      labels_onehot = torch.eye(NUM_CLASSES).to(device)
      labels_onehot = labels_onehot[labels]
      loss_is = cce_loss(F.softmax(outputs),labels_onehot)
      correct = correct + accuracy(outputs,labels,_,False)
      loss_list.append(loss_is.item())
    mean_loss = sum(loss_list)/len(loss_list)
    mean_acc = (correct/len(loader.dataset))*100
    if is_train:
      print("[%d/%d] Train Accuracy on actual samples : %f and Test Loss : %f"%(epochs,total_epochs,mean_acc,mean_loss))
    else:
      print("[%d/%d] Test Accuracy : %f and Test Loss : %f"%(epochs,total_epochs,mean_acc,mean_loss))
      print('---------------------------------------------------------------------')
  return mean_loss,mean_acc

dtype = torch.cuda.FloatTensor
torch.manual_seed(1341)
net = PreActResNet(PreActBlock, [2,2,2,2]).to(device)
opt = torch.optim.Adam(net.parameters(),lr = 0.001)
RESETCYCLE = 30
ALPHA = 0.7

total_epochs = 300
train_loss = []
train_acc = []
test_loss = []
test_acc = []
alpha_is = 0.7
for s in range(1,total_epochs + 1):
  a,b = train(net,s,trainloader,ALPHA)
  c,d = test(net,s,testloader,False)
  train_loss.append(a)
  train_acc.append(b)
  test_loss.append(c)
  test_acc.append(d)