import os
import torch
import  numpy as np
from torch._six import inf
from torch.utils.data import DataLoader, Dataset
import math
from matplotlib import pyplot as plt
import pandas as pd
from models import *
from net_temp import tp_net
import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def getSigma(eplison,delta,C,bitch):
    '''
        get sigma for Gaussian Differential Privacy
    '''
    sentive=2*C/bitch
    c=math.sqrt(2*math.log(1.25/delta))
    return c*sentive/eplison

def getSigma_up(eplison,delta,m,T,N,dp_rate):
    '''
        optimized upload sigma
    '''
    C=20
    L=N*dp_rate
    if T>L*math.sqrt(N):
        c = 2 * math.sqrt(math.log(1.25 / delta))
        sigma=(2*c*C*math.sqrt(T*T-L*L*N))/(m*N*eplison)
    else:
        sigma=0
    return sigma

def get_norm(X,model_name):
    '''
        Get the two-parametric number of each gradient
    '''
    Net=tp_net(model_name)
    Net.net.load_state_dict(X, strict=True)
    parameters=Net.net.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = list(parameters)
    parameter = list(p for p in parameters)
    total_norm = 0
    for p in parameter:
        param_norm = p.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    del Net
    return total_norm


def guidance_clip(guidance,model_name,C):
    '''
        guidance_clip
    '''
    Net = tp_net(model_name)
    Net.net.load_state_dict(guidance, strict=True)
    torch.nn.utils.clip_grad_norm_(Net.net.parameters(),C,2.0)
    guidance_clip=Net.net.state_dict()
    del Net
    return guidance_clip

def evaluate_acc_mnist(net, dev ,lossFun, global_parameter, testDataLoader):
    '''
        Test accuracy for mnist
    '''
    net.load_state_dict(global_parameter, strict=True)  # 将新的模型加载
    sum_accu = 0
    num = 0
    loss = 0
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss = lossFun(preds, label)
        preds = torch.argmax(preds, dim=1)  # 选择精确度最高的
        sum_accu += (preds == label).float().mean()
        num += 1
    print('accuracy: {}'.format(sum_accu / num))
    print('Loss:{}'.format(loss / num))

    return sum_accu / num, loss / num

def evaluate_acc_cifar(net, dev ,lossFun, global_parameter, testDataLoader):
    '''
        Test accuracy for cifar
    '''
    net.load_state_dict(global_parameter, strict=True)
    sum_accu = 0
    num = 0
    loss = 0
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss+=F.cross_entropy(preds,label,reduction='sum').item()
        preds=torch.argmax(preds,dim=1)
        sum_accu+=(preds==label).sum()
        num+=label.size(0)
    print('accuracy: {}%'.format(sum_accu / num *100.0))
    print('Loss:{}'.format(loss / num))
    return sum_accu / num, loss / num


def sava_model(net,save_path,model_name,num_comm,epoch,batchsize,learning_rate,num_of_clients,sample_rate,dp_rate):
    '''
        Save the trained model
    '''
    torch.save(net, os.path.join(save_path,'{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_spl{}_dprt{}'.format(model_name,num_comm, epoch,batchsize,learning_rate,num_of_clients,sample_rate,dp_rate)))


def draw_plot(x,y,model_name,num_comm,epoch,batchsize,learning_rate,num_of_clients,sample_rate,eplison,c):
    '''
        Draw line graphs
    '''
    plt.title('{}_accuracy'.format(model_name))
    plt.plot(x, y, c='green')
    plt.xlabel('communication rounds')
    plt.ylabel('Accuary')
    save_path='./result'
    # plt.savefig(os.path.join(save_path,'{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_spl{}_ep{}_c{}.png'.format(model_name,num_comm, epoch,batchsize,learning_rate,num_of_clients,sample_rate,eplison,c)))
    plt.show()

def save_accuracy(x,y,z,l,t,epoch,sample_rate,eplison,c):
    '''
        save result to xlsx
    '''
    save_path='./result'
    df=pd.DataFrame({'Epochs':x,'Accuracy':y,'sample_num':z,'Loss':l,'S+S_time':t})
    df.to_excel(os.path.join(save_path,"epcho{}_spl{}_ep{}_c{}.xlsx".format(epoch,sample_rate,eplison,c)),index=False)
