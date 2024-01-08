import paddle
import paddle.nn as nn


# 定义评论家网络结构
# DDPG这种方法与Q学习紧密相关，可以看作是连续动作空间的深度Q学习。
class Critic(nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        for m in self.__module__:
            if isinstance(m, nn.Linear):
                nn.initializer.XavierUniform(m.weight)

    def forward(self, state, action):
        x = paddle.concat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
