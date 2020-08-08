'''Train CIFAR10 with PyTorch'''
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.autograd import grad
import numpy as np
from models.lenet import *
from models.fcn_5 import *
from models.fcn_w import *
from models.fcn_d import *
from collections import OrderedDict

import torchvision
import torchvision.transforms as transforms


root = './results/'
if not os.path.exists(root):
	os.makedirs(root)

path_weights = root + 'net'
if not os.path.exists(path_weights):
	os.makedirs(path_weights)

path_log = root + 'log'
if not os.path.exists(path_log):
	os.makedirs(path_log)


path_data = './data/MNIST/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(net, batch_size, learning_rate, n, trian_epochs):
	transform = transforms.Compose([transforms.ToTensor()])
	print('Start loading dataset MNIST...')
	train_data = torchvision.datasets.MNIST(root=path_data, train=True, transform=transform, download=True)
	test_data = torchvision.datasets.MNIST(root=path_data,  train=False, transform=transform, download=True)
	print('Done loading dataset MNIST!')
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)

	for d, t in train_loader:
		train_X = Variable(d.to(device))
		train_y = Variable(t.to(device))

	for d, t in test_loader:
		test_X = Variable(d.to(device))
		test_y = Variable(t.to(device))

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

	net = net.to(device)
	if 'cuda' in device:
		print('Use CUDA!')
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True

	# create a stochastic gradient descent optimizer
	optimizer = optim.SGD(net.parameters(), lr=learning_rate)
	# create a loss function
	criterion = nn.CrossEntropyLoss()

	file_name = path_log + '/' + str(n)
	with open(file_name, "a") as f:
		data = ('bs=' + str(batch_size) + '\t' + 'lr=' + str(learning_rate) + '\t' + 'network=' + str(n))
		f.write(data)
		f.write('\n')

	epoch = 1
	test_arr_5 = []
	prev_loss = 100
	while True:
		net.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			net_out = net(data)
			loss = criterion(net_out, target)
			loss.backward()
			optimizer.step()

		net.eval()
		## train loss
		train_output = net(train_X)
		train_loss = criterion( train_output, train_y)
		train_loss_overall = np.sum( train_loss.data.cpu().numpy())
		_, predicted = train_output.max(1)
		train_correct = predicted.eq(train_y).sum().item()
		train_accuracy = 1. * train_correct / len(train_loader.dataset)


		## test loss
		test_output = net(test_X)
		test_loss = criterion(test_output, test_y)
		test_loss_overall = np.sum(test_loss.data.cpu().numpy())
		_, predicted = test_output.max(1)
		test_correct = predicted.eq(test_y).sum().item()
		test_accuracy = 1. * test_correct / len(test_loader.dataset)

		print(
			'Train Epoch: {}  \tTest Average loss: {:.6f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.6f}\tTrain Accuracy: {}/{} ({:.4f}%)'
				.format(epoch, test_loss, test_correct,
						len(test_loader.dataset), 100. * test_accuracy, train_loss, train_correct,
						len(train_loader.dataset),
						100. * train_accuracy))

		with open(file_name, "a") as f:
			data = (
				'Train Epoch: {}  \tTest Average loss: {:.6f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.6f}\tTrain Accuracy: {}/{} ({:.4f}%)'
					.format(epoch, test_loss, test_correct,
						len(test_loader.dataset), 100. * test_accuracy, train_loss, train_correct,
						len(train_loader.dataset),
						100. * train_accuracy))
			f.write(data)
			f.write('\n')

		# if epoch >= trian_epochs:
		if epoch >= trian_epochs:
			PATH = path_weights + '/' + str(n)
			torch.save(net.state_dict(), PATH)
			neuron_wise_measure(net, str(n), 'MNIST')
			break
		epoch += 1
		prev_loss = train_loss

if __name__ == "__main__":
	batch_size = [1000, 2000, 4000, 8000]
	learning_rate = [0.02, 0.04, 0.08, 0.16]
	n = 1
	epoch = 400
	for (bs, lr) in zip(batch_size, learning_rate):
		for i in range(20):
			print('bs=' + str(bs) + '\t' + 'lr=' + str(lr) + '\t' + 'NO.= ' + str(n))
			net = FCN_5()
			train(net, bs, lr, n, epoch)
			n = n+1