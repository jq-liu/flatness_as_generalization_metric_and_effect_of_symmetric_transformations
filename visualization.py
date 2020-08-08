import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
from matplotlib import pyplot as plt
from models.lenet import *
from models.fcn_5 import *
from models.fcn_w import *
from models.fcn_d import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_data = './data/MNIST/'


def load_dataset_MNIST(batch_size):
	print('Start loading dataset MNIST...')
	transform = transforms.Compose([transforms.ToTensor()])
	trainset_MNIST = torchvision.datasets.MNIST(root=path_data, train=True, download=True, transform=transform)
	trainloader_MNIST = torch.utils.data.DataLoader(trainset_MNIST, batch_size=batch_size, shuffle=True)
	testset_MNIST = torchvision.datasets.MNIST(root=path_data, train=False, download=True, transform=transform)
	testloader_MNIST = torch.utils.data.DataLoader(testset_MNIST, batch_size=batch_size, shuffle=False)
	print('Done loading dataset MNIST!')
	return trainloader_MNIST, testloader_MNIST


def crunch(net, coordinates, trainloader):
	net.to(device)
	if 'cuda' in device:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	w = get_weights(net)
	# d[0], d[1] and w are the same shape.
	d = get_direction(net)

	print('Start computing...')
	xcoordinates = coordinates[0]
	ycoordinates = coordinates[1]
	shape = (len(xcoordinates), len(ycoordinates))
	losses = -np.ones(shape=shape)
	accuracies = -np.ones(shape=shape)
	inds, coords = get_job_indices(losses, xcoordinates, ycoordinates)
	criterion = nn.CrossEntropyLoss()
	for count, ind in enumerate(inds):
		print('count = %d, ind = %d' % (count, ind))
		coord = coords[count]
		set_weights(net, w, d, coord)
		loss, acc = eval_loss(net, criterion, trainloader)
		losses.ravel()[ind] = loss
		accuracies.ravel()[ind] = acc
	return losses


def eval_loss(net, criterion, loader):
	for d, t in loader:
		train_X = Variable(d.to(device))
		train_y = Variable(t.to(device))
	net.eval()
	## train loss
	train_output = net(train_X)
	train_loss = criterion(train_output, train_y)
	train_loss_overall = np.sum(train_loss.data.cpu().numpy())
	_, predicted = train_output.max(1)
	train_correct = predicted.eq(train_y).sum().item()
	return train_loss_overall, 100. * train_correct / len(loader.dataset)


def get_weights(net):
	return [p.data for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
	if directions is None:
		for (p, w) in zip(net.parameters(), weights):
			p.data.copy_(w.type(type(p.data)))
	else:
		if len(directions) == 2:
			dx = directions[0]
			dy = directions[1]
			changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
		else:
			changes = [d * step for d in directions[0]]
		for (p, w, d) in zip(net.parameters(), weights, changes):
			p.data = w + torch.Tensor(d).type(type(w))


def get_direction(net):
	weights = get_weights(net)
	xdirection = [torch.randn(w.size()) for w in weights]
	normalize_directions(xdirection, weights)
	ydirection = [torch.randn(w.size()) for w in weights]
	normalize_directions(ydirection, weights)
	return [xdirection, ydirection]


def normalize_directions(direction, weights):
	for d, w in zip(direction, weights):
		if d.dim() <= 1:
			d.fill_(0)
		else:
			for di, we in zip(d, w):
				di.mul_(we.norm() / (di.norm() + 1e-10))


def get_job_indices(losses, xcoordinates, ycoordinates):
	inds = np.array(range(losses.size))
	inds = inds[losses.ravel() <= 0]
	if ycoordinates is not None:
		xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
		s1 = xcoord_mesh.ravel()[inds]
		s2 = ycoord_mesh.ravel()[inds]
		return inds, np.c_[s1, s2]
	else:
		return inds, xcoordinates.ravel()[inds]


def plot_2d_contour(op, coordinates, looses, vmin=0.1, vmax=10, vlevel=0.5):
	x = coordinates[0]
	y = coordinates[1]
	X, Y = np.meshgrid(x, y)
	Z = looses

	fig = plt.figure()
	CS = plt.contour(X, Y, Z, cmpa='summer', levels=np.arange(vmin, vmax, vlevel))
	plt.clabel(CS, inline=1, fontsize=8)
	fig.savefig(op + '_2Dcontour_' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
	plt.show()


def net_plt(x=1, y=1, num_axis=50):
	net = LeNet()

	name = 'lenet'

	if 'cuda' in device:
		print('Use CUDA!')
		net = torch.nn.DataParallel(net)
	else:
		print('No CUDA!')

	train_loader, test_loader = load_dataset_MNIST(60000)
	xcoordinates = np.linspace(-x, x, num_axis)
	ycoordinates = np.linspace(-y, y, num_axis)
	coordinates = [xcoordinates, ycoordinates]

	# state_dict = torch.load('/home/iai/user/jliu/visualize/' + name)
	# new_state_dict = OrderedDict()
	# for k, v in state_dict.items():
	# name_net = k[7:]  # remove module.
	# new_state_dict[name_net] = v
	# net.load_state_dict(new_state_dict)

	net.load_state_dict(torch.load('/home/iai/user/jliu/visualize/' + name))
	# net.load_state_dict(torch.load(net.name()))
	train_losses = crunch(net, coordinates, train_loader)
	plot_2d_contour(name + '_train', coordinates, train_losses, vmin=0.1, vmax=10, vlevel=0.5)
	np.savetxt(name + "_train_losses", train_losses)

	# state_dict = torch.load('/home/iai/user/jliu/visualize/' + name)
	# new_state_dict = OrderedDict()
	# for k, v in state_dict.items():
	# name_net = k[7:]  # remove module.
	# new_state_dict[name_net] = v
	# net.load_state_dict(new_state_dict)

	net.load_state_dict(torch.load('/home/iai/user/jliu/visualize/' + name))
	# net.load_state_dict(torch.load(net.name()))
	test_losses = crunch(net, coordinates, test_loader)
	plot_2d_contour(name + '_test', coordinates, test_losses, vmin=0.1, vmax=10, vlevel=0.5)
	np.savetxt(name + "_test_losses", test_losses)


if __name__ == '__main__':
	net_plt(1, 1, 20)


