import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch

def build_net(layer_shape, hid_activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hid_activation if j < len(layer_shape)-2 else hid_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2


class Policy_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Policy_Net, self).__init__()
		# layers = [state_dim] + list(hid_shape) + [action_dim]
		# self.P = build_net(layers, nn.ReLU, nn.Identity)
		for m in self.__module__:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
				m.bias.data.fill_(0.02)
		self.fc0 = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.LayerNorm(128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)

		self.fc1 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LayerNorm(256),
			nn.ReLU(),
		)

		self.fc2 = nn.Sequential(
			nn.Linear(256, action_dim)
		)

	def forward(self, s, replay_buffer):
		# mean = replay_buffer.state_mean()
		# std = replay_buffer.state_std()
		# s = s - mean
		# s = s / (std + 1e-8)
		# replay_buffer.state_mean()
		h = self.fc0(s/100000)
		h3 = self.fc1(h)
		logits = self.fc2(h3)
		# logits = self.P(s)
		# probs = act_mean / torch.sum(act_mean, dim=1)
		probs = F.softmax(logits, dim=1)
		return probs


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

	def state_mean(self):
		# 创建一个布尔掩码来识别非零元素
		mask = self.s != 0

		# 计算非零元素的均值和标准差
		mean = self.s[mask].mean()
		std = self.s[mask].std()

		# 对非零元素进行标准化
		self.s[mask] = (self.s[mask] - mean) / (std + 1e-8)
		return torch.mean(self.s, dim=0)

	def state_std(self):
		return torch.std(self.s, dim=0)

def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			s_next, r, done = env.step(a)
			total_scores += r
			s = s_next
	return total_scores/turns


#You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')