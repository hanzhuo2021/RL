from utils import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class TD3_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.policy_noise = 0.2*self.max_action
		self.noise_clip = 0.5*self.max_action
		self.tau = 0.005
		self.delay_counter = 0

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)
		self.memory = PrioritizedReplay(capacity=int(1e6))
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = np.array(state)
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			# state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
			# s, a, r, s_next, dw, idx, weights = self.memory.sample(self.batch_size)

			# Compute the target Q
			target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			'''↓↓↓ Target Policy Smoothing Regularization ↓↓↓'''
			smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
			target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
			'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~dw) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)
		td_error1 = target_Q.detach() - current_Q1  # ,reduction="none"
		td_error2 = target_Q.detach() - current_Q2
		prios = abs(((td_error1 + td_error2) / 2.0 + 1e-5).squeeze())
		# self.memory.update_priorities(idx, prios.data.cpu().numpy())
		# Compute critic loss, and Optimize the q_critic
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
		if self.delay_counter > self.delay_freq:
			# Update the Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			with torch.no_grad():
				for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = 0

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(np.array(s)).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(np.array(a)).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(np.array(s_next)).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


class PrioritizedReplay(object):
	"""
    Proportional Prioritization
    """

	def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
		self.alpha = alpha
		self.beta_start = beta_start
		self.beta_frames = beta_frames
		self.frame = 1  # for beta calculation
		self.capacity = capacity
		self.buffer = []
		self.pos = 0
		self.priorities = np.zeros((capacity,), dtype=np.float32)

	def beta_by_frame(self, frame_idx):
		"""
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
		return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

	def push(self, state, action, reward, next_state, done):
		# state = np.expand_dims(state, 0)
		# next_state = np.expand_dims(next_state, 0)

		max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

		if len(self.buffer) < self.capacity:
			self.buffer.append((state, action, reward, next_state, done))
		else:
			# puts the new data on the position of the oldes since it circles via pos variable
			# since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
			self.buffer[self.pos] = (state, action, reward, next_state, done)

		self.priorities[self.pos] = max_prio
		self.pos = (self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0

	def sample(self, batch_size):
		N = len(self.buffer)
		if N == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.pos]

		# calc P = p^a/sum(p^a)
		probs = prios ** self.alpha
		P = probs / probs.sum()

		# gets the indices depending on the probability p
		indices = np.random.choice(N, batch_size, p=P)
		samples = [self.buffer[idx] for idx in indices]

		beta = self.beta_by_frame(self.frame)
		self.frame += 1

		# Compute importance-sampling weight
		weights = (N * P[indices]) ** (-beta)
		# normalize weights
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)

		states, actions, rewards, next_states, dones = zip(*samples)
		states = np.vstack(states)
		rewards = np.vstack(rewards)
		next_states = np.vstack(next_states)
		dones = np.vstack(dones)
		states = torch.from_numpy(np.array(states, dtype=np.float32)).to("cuda")
		actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to("cuda")
		rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to("cuda")
		next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to("cuda")
		dones = torch.from_numpy(np.array(dones, dtype=np.bool_)).to("cuda")
		return states, actions, rewards, next_states, dones, indices, weights

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = abs(prio)

	def __len__(self):
		return len(self.buffer)


