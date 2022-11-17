from models import *

class tp_net(object):
    def __init__(self,model_name):
        if model_name == 'mnist_2nn':
            self.net = Mnist_2NN()  # Initialising the network 2NN
        elif model_name == 'mnist_cnn':
            self.net = Mnist_CNN()  # Initialising the network CNN
        elif model_name == 'cifar_cnn':
            self.net = Cifar_CNN(10)
        elif model_name == 'cifar_resnet':
            self.net = Cifar_ResNet()
