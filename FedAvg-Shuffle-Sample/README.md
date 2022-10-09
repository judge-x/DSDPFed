# DSDPFed

### environment

conda create -n ***

pip install -r requirements.txt

##### PyTorch-version

1.python3.7.6

2.pytorch1.4.0

both of them run on GPU

### prepare data sets

You are supposed to prepare the data set by yourself. MNIST can be downloaded on http://yann.lecun.com/exdb/mnist/, and CIFAR-10 can be downloaded on http://www.cs.toronto.edu/~kriz/cifar.html. These data sets should be put into /data/MNIST and /data/cifar when the download is finished.
And cifar dataset will be download automaticly.
### usage

Run the code

with mnist 
```asp
python train.py -nc 100 -cf 0.5 -E 5 -B 10 -mn mnist_cnn  -ncomm 50 -iid 0 -lr 0.01 -vf 1  -g 0 -dp 0
```

or with cifar
```asp
python train.py -nc 100 -cf 0.5 -E 5 -B 10 -mn cifar_cnn  -ncomm 100 -iid 1 -lr 0.01 -vf 1  -g 0 -dp 1 -drs 1 -dp_rt 1 -ep 1
```

The parameter setting reference is as follows:
```asp
#base parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')       #gpu数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')   #客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')      #每次选择运行的客户端的百分比
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')     #本地迭代的轮数
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')   #SGD梯度下降选择的块的大小
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')    #选择训练的模型
parser.add_argument('-nclass','--num_classes', type=int, default=10, help="number of classes")      #分类的数量
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, use value from origin paper as default")    #定义学习速率
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")     #模型通信验证的频率
parser.add_argument('-sf', '--save_freq', type=int, default=1, help='global model save frequency(of communication)')   #全局模型保存频率(通信)
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')    #通信次数
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')     #检查点的保存地址
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')   #划分客户端数据


#guidance sparse
parser.add_argument('-spr', '--sample_rate', type=float, default=.1, help='the sample rate')    #稀疏化采样概率
parser.add_argument('-c', '--thr rule', type=float, default=.2, help='the rule of value-sorted client')     #稀疏化的裁切阈值


#privacy
parser.add_argument('-drs','--dp_rt_switch',type=int,default=0,help='input dp_rt or not')       #是否采用DPSFed
parser.add_argument('-dp', '--dp_switch', type=int, default=1, help='the dp run or not')       #是否使用DP
parser.add_argument('-ep', '--epsilon', type=float, default=1, help='the privacy burget')       #隐私预算
parser.add_argument('-de', '--delta', type=float, default=0.01, help='the slack')       #松弛度
parser.add_argument('-dp_rt', '--dp_rate', type=float, default=0.6, help='the probability of differential rate')        #dp的概率

'''
