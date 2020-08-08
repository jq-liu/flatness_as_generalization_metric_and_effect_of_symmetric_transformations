'''Fully connected networks with different width.'''
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FCN_w_1(nn.Module):
	def __init__(self):
		super(FCN_w_1, self).__init__()
		self.fc1 = nn.Linear(784, 50)
		init.xavier_normal_(self.fc1.weight.data)
		init.zeros_(self.fc1.bias.data)
		self.fc2 = nn.Linear(50, 50)
		init.xavier_normal_(self.fc2.weight.data)
		init.zeros_(self.fc2.bias.data)
		self.fc3 = nn.Linear(50, 50)
		init.xavier_normal_(self.fc3.weight.data)
		init.zeros_(self.fc3.bias.data)
		self.fc4 = nn.Linear(50, 50)
		init.xavier_normal_(self.fc4.weight.data)
		init.zeros_(self.fc4.bias.data)
		self.fc5 = nn.Linear(50, 10)
		init.xavier_normal_(self.fc5.weight.data)
		init.zeros_(self.fc5.bias.data)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))
		out = self.fc5(out)
		return out
	def name(self):
		return "fcn-w_1"

class FCN_w_2(nn.Module):
	def __init__(self):
		super(FCN_w_2, self).__init__()
		self.fc1 = nn.Linear(784, 75)
		init.xavier_normal_(self.fc1.weight.data)
		init.zeros_(self.fc1.bias.data)
		self.fc2 = nn.Linear(75, 75)
		init.xavier_normal_(self.fc2.weight.data)
		init.zeros_(self.fc2.bias.data)
		self.fc3 = nn.Linear(75, 75)
		init.xavier_normal_(self.fc3.weight.data)
		init.zeros_(self.fc3.bias.data)
		self.fc4 = nn.Linear(75, 75)
		init.xavier_normal_(self.fc4.weight.data)
		init.zeros_(self.fc4.bias.data)
		self.fc5 = nn.Linear(75, 10)
		init.xavier_normal_(self.fc5.weight.data)
		init.zeros_(self.fc5.bias.data)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))
		out = self.fc5(out)
		return out
	def name(self):
		return "fcn-w_2"

class FCN_w_3(nn.Module):
	def __init__(self):
		super(FCN_w_3, self).__init__()
		self.fc1 = nn.Linear(784, 100)
		init.xavier_normal_(self.fc1.weight.data)
		init.zeros_(self.fc1.bias.data)
		self.fc2 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc2.weight.data)
		init.zeros_(self.fc2.bias.data)
		self.fc3 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc3.weight.data)
		init.zeros_(self.fc3.bias.data)
		self.fc4 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc4.weight.data)
		init.zeros_(self.fc4.bias.data)
		self.fc5 = nn.Linear(100, 10)
		init.xavier_normal_(self.fc5.weight.data)
		init.zeros_(self.fc5.bias.data)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))
		out = self.fc5(out)
		return out
	def name(self):
		return "fcn-w_3"


