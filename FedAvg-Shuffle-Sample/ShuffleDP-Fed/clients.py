import torch.nn.utils
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader,Dataset,Subset
from get_Data import GetDataSet
from utils import *
from sample import *
from gaussian_Mechanism import GaussianMechanism
import numpy as np
import math
class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, dataset_name,localEpoch, localBatchSize,model_name, Net, lossFun, opti, global_parameters,dp,epsilon,delta,drs,dr):

        # Importing models based on global variables
        Net.load_state_dict(global_parameters, strict=True)
        C=1  #max_norm
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        optimizer=opti
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)

                preds =Net(data)

                loss = lossFun(preds, label)

                torch.nn.utils.clip_grad_norm_(Net.parameters(), C, 2)

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

        local_guidance=Net.state_dict()

        #get contribute
        for key, var in local_guidance.items():

            local_guidance[key] = var.cuda() - global_parameters[key].cuda()

        if dp==1 and drs==0:
            #Dynamic rate
            dp_rate=1-1/(1+math.exp(-1*float(get_norm(local_guidance,model_name))/float(get_norm(global_parameters,model_name))))
            #Choose to add Gaussian noise and add Gaussian differential privacy if the switch is equal to 1 and the dp_rate probability is sampled
            if np.random.choice([0,1],p=[(1-dp_rate),dp_rate])==1:
                sigma=getSigma(epsilon,delta,C,len(self.train_ds))
                for key,var in enumerate(local_guidance):
                    local_guidance[var]=GaussianMechanism(local_guidance[var],sigma,True)    #if cpu,change it for False

        elif dp==1 and drs==1:
            #input dp_rate
            dp_rate=dr
            if np.random.choice([0,1],p=[(1-dp_rate),dp_rate])==1:
                sigma=getSigma(epsilon,delta,C,len(self.train_ds))
                for key, var in enumerate(local_guidance):
                    local_guidance[var] = GaussianMechanism(local_guidance[var], sigma,True)

        return local_guidance



class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        if self.data_set_name=='mnist':
            mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

            test_data = torch.tensor(mnistDataSet.test_data)
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
            self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label

            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_label = np.argmax(local_label, axis=1)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone

        elif self.data_set_name=='cifar':
            cifarDataSet = GetDataSet(self.data_set_name, self.is_iid)

            train_data=cifarDataSet.train_data
            test_data=cifarDataSet.test_data

            test_idxs=[i for i in range(len(test_data))]
            self.test_data_loader=DataLoader(DatasetSplit(test_data,test_idxs), batch_size=100, shuffle=False)

            if self.is_iid:
                num_items = int(len(train_data) / self.num_of_clients)
                dict_users, all_idxs = {}, [i for i in range(len(train_data))]
                for i in range(self.num_of_clients):
                    dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                    all_idxs = list(set(all_idxs) - dict_users[i])
                    # someone = client(all_idxs[i*num_items,(i+1)*num_items], self.dev)
                    someone =client(DatasetSplit(train_data, dict_users[i]),self.dev)
                    self.clients_set['client{}'.format(i)] = someone
            else:
                exit('cifar is only for niid')


if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


