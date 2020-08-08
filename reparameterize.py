import torch
import random
from random import choice
import os
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.lenet import *
from models.fcn_5 import *
from models.fcn_w import *
from models.fcn_d import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


path_net = './results/net/'
path_net_re = './results/net_re'
if not os.path.exists(path_net_re):
	os.makedirs(path_net_re)


class fcn:
	def reparameterize_neuron(net):
		layer = random.randint(2, 3)
		neuron = -1
		ratio = random.randint(2, 20)
		for layer_index, param_l in enumerate(net.parameters()):
			if layer_index == 2*layer:
				# print(param_l[neuron])
				count = param_l.size()[0]
				neuron = random.randint(0, count - 1)
				param_l[neuron] *= ratio
				# print(param_l[neuron])
			elif layer_index == 2*layer + 1:
				param_l[neuron] *= ratio
			elif layer_index == 2*layer + 2:
				for param_n in param_l:
					param_n[neuron] /= (1.0*ratio)
		return net


class lenet:
	def reparameterize_neuron(net):
		layer = random.randint(0, 1)
		ratio = random.randint(2, 20)
		neuron = -1
		print('layer = ' + str(layer))
		for layer_index, param_l in enumerate(net.parameters()):
			# print('index = ' + str(layer_index))
			# print(param_l.size())
			if layer_index == 2*layer:
				count = param_l.size()[0]
				neuron = random.randint(0, count - 1)
				print('neuron = ' + str(neuron))
				param_l[neuron] *= ratio
			elif layer_index == 2*layer+1:
				param_l[neuron] *= ratio
			elif layer_index == 2*layer+2:
				if layer == 1:
					temp = int(param_l.size()[1]/count)
					for param_n in param_l:
						for i in range(neuron*temp, (neuron+1)*temp):
							param_n[i] /= (1.0*ratio)
				else:
					for param_n in param_l:
						param_n[neuron] /= (1.0*ratio)
		return net







