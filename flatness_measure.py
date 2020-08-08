import os
import torch
import random
import numpy as np
import torchvision
from collections import OrderedDict

import torch.nn as nn
from torch.autograd import Variable
from train_networks import load_dataset_MNIST
from torch.autograd import grad
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils import data
np.set_printoptions(threshold=np.inf)



path_net = './results/net/'
path_flatness = './results/flatness'
if not os.path.exists(path_flatness):
	os.makedirs(path_flatness)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# calculate the flatness for networks with only fully connected layers.
def neuron_wise_measure_for_FCN(net, net_name):
	train_loader, test_loader = load_dataset_MNIST(60000)

	for batch_idx, (data, target) in enumerate(train_loader):
		print('Start calculation for flatness!')
	net = net.to(device)
	data, target = data.to(device), target.to(device)
	data, target = Variable(data), Variable(target)
	criterion = nn.CrossEntropyLoss()

	net_out = net(data)
	loss = criterion(net_out, target)
	net_measure = []
	for layer_i, layer_p in enumerate(net.parameters()):

		if layer_i == 0:
			continue

		# skip the bias
		if layer_i % 2 == 1:
			continue

		shape = layer_p.shape
		layer_g = grad(loss, layer_p, create_graph=True, retain_graph=True)[0]

		layer_measure = []
		print('.')
		for column in range(shape[1]):
			print(str(column))
			column_p = layer_p[:, column]
			hessian = []
			for neuron_g in layer_g:
				grad2rd = grad(neuron_g[column], layer_p, retain_graph=True)
				hessian.append(grad2rd[0][:, column].data.cpu().numpy().flatten())
			weight_ij = np.reshape(column_p.detach().cpu().numpy(), (1, shape[0]))
			m = weight_ij.dot(np.array(hessian)).dot(np.transpose(weight_ij))[0][0]
			layer_measure.append(m)
		net_measure.append(np.array(layer_measure))
	np.save(path_flatness + '/' + net_name, net_measure)


# calculate the flatness for networks with fully connected layers and convolutional layers - LeNet 5 in this experiment.
def neuron_wise_measure_for_LeNet(net, net_name):
	train_loader, test_loader = load_dataset_MNIST(60000)
	for batch_idx, (data, target) in enumerate(train_loader):
		print('Start to calculate the flatness of ' + str(net_name) + '-th network.')
	net = net.to(device)
	data, target = data.to(device), target.to(device)
	data, target = Variable(data), Variable(target)
	criterion = nn.CrossEntropyLoss()

	net_out = net(data)
	loss = criterion(net_out, target)
	net_measure = []
	for layer_i, layer_p in enumerate(net.parameters()):

		# skip the first layer
		if layer_i == 0:
			continue
		# skip the bias
		if layer_i % 2 == 1:
			continue
		print("Calculate the flatness of the " + str(layer_i) + "-th layer.")
		shape = layer_p.shape
		layer_g = grad(loss, layer_p, create_graph=True, retain_graph=True)[0]

		if layer_i == 2:
			net_measure.append(con_flatness(layer_g, layer_p))
			continue

		layer_measure = []
		print('.')
		for column in range(shape[1]):
			if column % 10 == 0:
				print(str(column))
			column_p = layer_p[:, column]
			hessian = []
			for neuron_g in layer_g:
				kernel = torch.reshape(neuron_g[column], [-1])
				for index in range(len(kernel)):
					grad2rd = grad(kernel[index], layer_p, retain_graph=True)
					hessian.append(grad2rd[0][:, column].data.cpu().numpy().flatten())
			weight_ij = np.reshape(column_p.detach().cpu().numpy(), (1, -1))
			m = weight_ij.dot(np.array(hessian)).dot(np.transpose(weight_ij))[0][0]
			layer_measure.append(m)
		net_measure.append(np.array(layer_measure))
		del layer_measure
	np.save(path_flatness + '/' + net_name, net_measure)
	del net
	del net_out
	del net_measure

# calculate the flatness for the second layer in LeNet-5 trained with MNIST.
def con_flatness(layer_g, layer_p):
	index = np.load('./index.npy', allow_pickle=True)
	layer_measure = []
	for kernel_i in range(6):
		print('.')
		for column in range(144):
			print(str(column))
			if column % 12 == 5 or column % 12 == 6 or column % 12 ==7:
				layer_measure.append(layer_measure[column-1])
				continue
			if column >= 60 and column <= 95:
				layer_measure.append(layer_measure[column-12])
				continue
			temp = index[column]
			weights = []
			hessian = []
			for neuron_i, (neuron_p, neuron_g) in enumerate(zip(layer_p, layer_g)):
				kernel_p = neuron_p[kernel_i]
				kernel_g = neuron_g[kernel_i]
				for postion in temp:
					x = int(postion/5)
					y = postion%5
					weights.append(kernel_p[x][y].item())
					grad2rd = grad(kernel_g[x][y], layer_p, retain_graph=True)[0]
					h_row = []
					for g in grad2rd:
						for i in temp:
							x_i = int(i/5)
							y_i = i%5
							h_row.append(g[kernel_i][x_i][y_i].item())
					hessian.append(h_row)
			weights = np.reshape(np.array(weights), (1, -1))
			m = weights.dot(np.array(hessian)).dot(np.transpose(weights))[0][0]
			layer_measure.append(m)
	return np.array(layer_measure)




