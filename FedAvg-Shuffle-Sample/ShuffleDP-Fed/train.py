import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import *
from clients import ClientsGroup
from shuffler import shuffle
from server import update
from utils import *
from sample import sample_select
from sample_former import  sample_select_former
import time


#base parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')   #客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')      #每次选择运行的客户端的百分比
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')     #本地迭代的轮数
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')   #SGD梯度下降选择的块的大小
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')    #选择训练的模型
parser.add_argument('-nclass','--num_classes', type=int, default=10, help="number of classes")
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                    use value from origin paper as default")    #定义学习速率
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")     #模型通信验证的频率
parser.add_argument('-sf', '--save_freq', type=int, default=1, help='global model save frequency(of communication)')   #全局模型保存频率(通信)
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')    #通信次数
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')     #检查点的保存地址
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')   #划分客户端数据

#guidance sparse
parser.add_argument('-spr', '--sample_rate', type=float, default=.6, help='the sample rate')
parser.add_argument('-c', '--thr rule', type=float, default=.2, help='the rule of value-sorted client')


#privacy
parser.add_argument('-dp', '--dp_switch', type=int, default=1, help='the dp run or not')
parser.add_argument('-ep', '--epsilon', type=float, default=1, help='the privacy burget')
parser.add_argument('-de', '--delta', type=float, default=0.01, help='the slack')
parser.add_argument('-drs','--dp_rt_switch',type=int,default=0,help='input dp_rt or not')
parser.add_argument('-dp_rt', '--dp_rate', type=float, default=0.6, help='the probability of differential rateS')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    # test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    #将GPU运行转变为CPU运行
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   #choose gpu to train

    # Initialize the network
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
        dataset_name = 'mnist'
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
        dataset_name = 'mnist'
    elif args['model_name'] == 'cifar_cnn':
        net = Cifar_CNN(args['num_classes'])
        dataset_name = 'cifar'
    elif args['model_name'] == 'cifar_resnet':
        net = Cifar_ResNet()
        dataset_name = 'cifar'
    else:
        exit('no such model')
    print(net)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net).cuda()

    net = net.to(dev)

    loss_func = F.cross_entropy

    opti = optim.SGD(net.parameters(), lr=args['learning_rate'],momentum=0.9,weight_decay=5e-4)

    myClients = ClientsGroup(dataset_name, args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    #xy axis for plotting, indicating number of rounds and accuracy respectively
    x_numcom=[]
    y_accuracy=[]
    Z_sample=[]
    L_sample=[]
    Time=[]
    i=0
    for i in range(int(args['num_comm']/args['val_freq'])):
        x_numcom.append(i+1)


    global_parameter = {}
    for key, var in net.state_dict().items():
        global_parameter[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        local_parameters=[]
        init_parameter = None
        # norm_sum=0

        # print(opti_drease.get_last_lr()[0])

        for client in tqdm(clients_in_comm):
            local_parameter = myClients.clients_set[client].localUpdate(dataset_name,args['epoch'], args['batchsize'],args['model_name'], net, loss_func, opti, global_parameter,args['dp_switch'],
                                                                        args['epsilon'],args['delta'],args['dp_rt_switch'],args['dp_rate'])
            local_parameters.append(local_parameter)
            # norm_sum+=norm

        time_start=time.time()
        #select from global
        local_parameters=sample_select(local_parameters,args['sample_rate'],args['model_name'],args['thr rule'])

        #shuffle
        shuffle(local_parameters)
        time_end=time.time()
        
        print("select+shuffle的时间是:")

        print(time_end-time_start)
        Time.append(time_end-time_start)

        #get global_guidance
        global_parameter=update(local_parameters,global_parameter,init_parameter,args['num_of_clients'],0.01)

        # opti_drease.step()

        if (i + 1) % args['val_freq'] == 0:
            #save accuracy
            if dataset_name=='mnist':
                print("evaluate mnist")
                accuracy_temp, loss = evaluate_acc_mnist(net, dev, loss_func, global_parameter, testDataLoader)
            else:
                accuracy_temp, loss = evaluate_acc_cifar(net, dev, loss_func, global_parameter, testDataLoader)

            y_accuracy.append(accuracy_temp.item())
            #save_sample_num
            Z_sample.append(len(local_parameters))
            #save loss
            if dataset_name=='mnist':
                L_sample.append(loss.item())
            else:
                L_sample.append(loss)

            #save norm
            # N_norm.append(norm_sum/args['num_of_clients'])


        if (i+1)==args['num_comm']:
            # save accuracy to local
            save_accuracy(x_numcom,y_accuracy,Z_sample,L_sample,Time,(i+1),args['sample_rate'],args['epsilon'],args['thr rule'])
