import torch
import random

def sample_select_former(local_guidances,former_guidances,rate,sum_num):
    '''
        梯度抽样
    '''
    sample_num=int(rate*sum_num)
    sample_guidances=random.sample(local_guidances,sample_num)
    former_guidances_two=float(get_norm_2(former_guidances))
    thr=get_thr(sample_guidances,former_guidances_two,int(sample_num*0.2))
    selected_guidances=select_guidances(local_guidances,former_guidances_two,thr)
    print(len(selected_guidances))
    return selected_guidances

def get_thr(sample_guidances,former_guidances_two,k):
    '''
        返回抽样的阈值
    '''
    list=[]
    for x in sample_guidances:
        list.append(abs(float(get_norm_2(x))-former_guidances_two))

    list.sort(reverse=False)
    return list[k-1]

def select_guidances(local_guidances,former_guidances_two,thr):
    '''
        采样梯度
    '''
    selected_guidances=[]
    for x in local_guidances:
        if abs(float(get_norm_2(x))-former_guidances_two)<thr:
            selected_guidances.append(x)
        else:
            pass
    return selected_guidances



def get_norm_2(X):
    '''
        获取每一个梯度的二范数
    '''
    for key,var in enumerate(X):
        norm_2=torch.norm(X[var],2)
    return norm_2