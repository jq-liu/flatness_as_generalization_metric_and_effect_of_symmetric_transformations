'''Fully connected networks with different depth.'''
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class FCN_d_1(nn.Module):
	def __init__(self):
		super(FCN_d_1, self).__init__()
		self.fc1 = nn.Linear(784, 100)
		init.xavier_normal_(self.fc1.weight.data)
		init.zeros_(self.fc1.bias.data)
		self.fc2 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc2.weight.data)
		init.zeros_(self.fc2.bias.data)
		self.fc3 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc3.weight.data)
		init.zeros_(self.fc3.bias.data)
		self.fc4 = nn.Linear(100, 10)
		init.xavier_normal_(self.fc4.weight.data)
		init.zeros_(self.fc4.bias.data)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = self.fc4(out)
		return out
	def name(self):
		return "fcn_d_1"

class FCN_d_2(nn.Module):
	def __init__(self):
		super(FCN_d_2, self).__init__()
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
		return "fcn_d_2"

class FCN_d_3(nn.Module):
	def __init__(self):
		super(FCN_d_3, self).__init__()
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
		self.fc5 = nn.Linear(100, 100)
		init.xavier_normal_(self.fc5.weight.data)
		init.zeros_(self.fc5.bias.data)
		self.fc6 = nn.Linear(100, 10)
		init.xavier_normal_(self.fc6.weight.data)
		init.zeros_(self.fc6.bias.data)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))
		out = F.relu(self.fc5(out))
		out = self.fc6(out)
		return out
	def name(self):
		return "fcn_d_3"

