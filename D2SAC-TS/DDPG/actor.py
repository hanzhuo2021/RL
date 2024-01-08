import numpy as np
import paddle
import paddle.nn as nn
from paddle.distribution import Normal


# 定义演员网络结构
# 为了使DDPG策略更好地进行探索，在训练时对其行为增加了干扰。 原始DDPG论文的作者建议使用时间相关的 OU噪声 ，
# 但最近的结果表明，不相关的均值零高斯噪声效果很好。 由于后者更简单，因此是首选。
class Actor(nn.Layer):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.noisy = Normal(0, 0.2)
        self.max_action = max_action
        for m in self.__module__:
            if isinstance(m, nn.Linear):
                nn.initializer.XavierUniform(m.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        # x = self.max_action * x
        # 根据动作设定，映射至[0,1]
        # x = (x + 1.) / 2.
        return x

    def select_action(self, epsilon, state):
        state = state / np.linalg.norm(state)
        state = paddle.to_tensor(state, dtype="float32").unsqueeze(0)
        action = self.forward(state).squeeze()
        noisy = epsilon * self.noisy.sample([1]).squeeze(0)
        action += noisy
        return paddle.clip(action, -1, 1).numpy()
