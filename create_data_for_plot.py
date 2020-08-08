import torch
import os
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.lenet import *
from models.fcn_5 import *
from models.fcn_w import *
from models.fcn_d import *
import torchvision
import torchvision.transforms as transforms

log_interval = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_data = './plot/data/'

path_flatness = './flatness/'



def generalization_gap(net):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root='./data/MNIST/', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root='./data/MNIST/',  train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)

    for d, t in train_loader:
        train_X = Variable(d.to(device))
        train_y = Variable(t.to(device))

    for d, t in test_loader:
        test_X = Variable(d.to(device))
        test_y = Variable(t.to(device))

    loss = nn.CrossEntropyLoss()

    ## test loss
    test_output = net(test_X)
    test_loss = loss(test_output, test_y)
    test_loss_overall = np.sum(test_loss.data.cpu().numpy()) * 6
    _, predicted = test_output.max(1)
    test_correct = predicted.eq(test_y).sum().item() * 6
    # test_accuracy = 1. * train_correct / len(test_loader.dataset)
    # print("Six times test loss calculated: {}".format(test_loss_overall))
    # print("Test correct: {}".format(test_correct))

    ## train loss
    train_output = net(train_X)
    train_loss = loss( train_output, train_y)
    train_loss_overall = np.sum( train_loss.data.cpu().numpy() )
    _, predicted = train_output.max(1)
    train_correct = predicted.eq(train_y).sum().item()
    # train_accuracy = 1. * train_correct / len(train_loader.dataset)
    # print( train_loss_overall)
    # print( "Train loss calculated: {}".format(train_loss_overall) )
    # print("Train correct: {}".format(train_correct))

    generalization_error = test_loss_overall - train_loss_overall
    generalization_corre = train_correct - test_correct
    # print("Generalization error calculated: {}".format(generalization_error))
    return generalization_error, generalization_corre, train_loss_overall, test_loss.data.cpu().numpy()


def layer_max(flatness_neuron):
	l_max = []
	n = flatness_neuron.shape[0]
	for i in range(n):
		m = np.amax(np.array(flatness_neuron[i]))
		l_max.append(m)
	return l_max


def layer_sum(flatness_neuron):
	l_sum = []
	n = flatness_neuron.shape[0]
	for i in range(n):
		m = np.sum(np.array(flatness_neuron[i]))
		l_sum.append(m)
	return l_sum


def network_max(flatness_neuron):
	l_max = layer_max(flatness_neuron)
	return np.amax(l_max)


def network_sum(flatness_neuron):
	l_sum = layer_sum(flatness_neuron)
	return np.sum(l_sum)


def cal_l(method, number_of_networks):
	l = []
	for i in range(1, number_of_networks):
		flatness_neuron = np.load(path_flatness  + str(i) + '.npy', allow_pickle=True)
		if method == 'max':
			f = layer_max(flatness_neuron)
		else:
			f = layer_sum(flatness_neuron)

		if len(l) == 0:
			for j in range(len(f)):
				temp = []
				temp.append(f[j])
				l.append(temp)
		else:
			for j in range(len(f)):
				l[j].append(f[j])

	for i in range(len(l)):
		np.save(path_data + 'l_' + method + '_' + str(i), l[i])


def cal_n(method, number_of_networks):
	n = []
	for i in range(1, number_of_networks):
		flatness_neuron = np.load(path_flatness + str(i) + '.npy', allow_pickle=True)
		if method == 'max':
			n.append(network_max(flatness_neuron))
		else:
			n.append(network_sum(flatness_neuron))
	np.save(path_data + 'n_' + method, n)


if __name__ == "__main__":
	number_of_networks = 61

	# calculate the maximun and sum of flatness for each layer.
	cal_l('max', number_of_networks)
	cal_l('sum', number_of_networks)
	# calculate the maximun and sum of flatness for the whole network.
	cal_n('max', number_of_networks)
	cal_n('sum', number_of_networks)

	# calculate the generalization error for networks.
	# net_path contains trained networks.
	net_path = ''
	gen_loss = []
	gen_corr = []
	train_loss = []
	test_loss = []
	for i in range(1, number_of_networks):
		net = LeNet()
		net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
		gen_l, gen_c, train_l, test_l = data(net)
    	gen_loss.append(gen_l)
    	train_loss.append(train_l)
    	test_loss.append(test_l)
    	gen_corr.append(gen_c)
	np.save(path_data + 'gen_loss', np.array(gen_loss))
	np.save(path_data + 'gen_corr', np.array(gen_corr))
	np.save(path_data + 'train_loss', np.array(train_loss))
	np.save(path_data + 'test_loss', np.array(test_loss))











