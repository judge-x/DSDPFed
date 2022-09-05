import torch
import random
from utils import *

def sample_select(local_guidances,rate,model_name,c):
    '''
        梯度抽样
    '''
    sample_num=int(rate*len(local_guidances))
    sample_guidances=random.sample(local_guidances,sample_num)

    #get thr with sample_rate,default:0.5
    thr=get_thr(sample_guidances,int(sample_num*c),model_name)
    #get selected guidances
    selected_guidances=select_guidances(local_guidances,thr,model_name)

    #print the number of selected guidances in one epchos
    print(len(selected_guidances))

    return selected_guidances

def get_thr(sample_guidances,k,model_name):
    '''
        返回抽样的阈值
    '''
    list=[]
    for x in sample_guidances:
        list.append(float(get_norm(x,model_name)))
    list.sort(reverse=True)
    return list[k-1]

def select_guidances(local_guidances,thr,model_name):
    '''
        采样梯度
    '''
    selected_guidances=[]
    for x in local_guidances:
        if float(get_norm(x,model_name))>=thr:
            selected_guidances.append(x)
        else:
            pass
    return selected_guidances

