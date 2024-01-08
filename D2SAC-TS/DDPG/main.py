import time

import numpy as np
import paddle
import paddle.nn.functional as F
from tqdm import tqdm
from visualdl import LogWriter

import pandas as pd
from actor import Actor
from critic import Critic
from ilabEnv.envYiLai import Environment
from memory import Memory
from matplotlib import pyplot as plt
import sys
max_float = sys.float_info.max

print(paddle.get_device())
# paddle.set_device('gpu:0')
train_begin_time = time.strftime("%Y-%m-%d-%H-%M-%S")

log_dir = './' + train_begin_time + '/log'
model_dir = './' + train_begin_time + '/model'

writer = LogWriter(logdir=log_dir)

print('visualdl --logdir ' + log_dir + ' --port 4523')
# visualdl --logdir ./log-2022-12-08-16-07-51 --port 4523


# 定义超参数
explore = 50000
epsilon = 1
gamma = 0.99
tau = 0.001
lr = 1e-4

# 定义环境、实例化模型
env = Environment()
# state_dim = 44
# state_dim = 66
# state_dim = 88
state_dim = 110
action_dim = 1
max_action = 1

# 动作网络与目标动作网络
actor = Actor(state_dim, action_dim, max_action)
actor_target = Actor(state_dim, action_dim, max_action)

# 值函数网络与目标值函数网络
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)

# 定义优化器
critic_optim = paddle.optimizer.Adam(parameters=critic.parameters(), learning_rate=3 * lr)
actor_optim = paddle.optimizer.Adam(parameters=actor.parameters(), learning_rate=lr)

# 初始化经验池
memory_replay = Memory(50000)
begin_train = False
batch_size = 128

learn_steps = 0
epochs = 2000


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 定义软更新的函数
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.set_value(target_param * (1.0 - tau) + param * tau)

csv_field = {}
makespan_field = {"best": [], "final": []}
# 运行十次
for num in range(1):
    episode_reward_list = []
    makespan = max_float
    for i in range(10):
        with tqdm(total=int(epochs / 10), desc='Iteration %d' % i) as pbar:
            for epoch in range(int(epochs / 10)):
                # # 训练循环
                # for epoch in range(0, epochs):
                # 状态初始化
                state = env.reset()
                # 智能体与环境交互一个回合的回合总奖励
                episode_reward = 0
                for time_step in range(10000):
                    action = actor.select_action(epsilon, state)
                    next_state, reward, done = env.step(action)
                    # reward *= 5

                    if epsilon > 0.01:
                        epsilon -= 10 / explore

                    episode_reward += reward
                    memory_replay.add((state, next_state, action, reward))
                    state = next_state
                    if done:
                        break
                    if memory_replay.size() > 5000:
                        learn_steps += 1
                        if not begin_train:
                            print('train begin!')
                            begin_train = True
                        # 从缓存容器中采样
                        experiences = memory_replay.sample(batch_size, False)
                        batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

                        batch_state = batch_state / np.linalg.norm(batch_state)
                        batch_next_state = batch_next_state / np.linalg.norm(batch_next_state)

                        batch_state = paddle.to_tensor(batch_state, dtype="float32")
                        batch_next_state = paddle.to_tensor(batch_next_state, dtype="float32")
                        batch_action = paddle.to_tensor(batch_action, dtype="float32").unsqueeze(1)
                        batch_reward = paddle.to_tensor(batch_reward, dtype="float32")

                        # 计算目标网络q值
                        q_next = critic_target(batch_next_state, actor_target(batch_next_state))
                        q_target = batch_reward + gamma * q_next

                        # 计算当前网络q值
                        q_eval = critic(batch_state, batch_action)

                        # 计算值网络的损失函数
                        critic_loss = F.mse_loss(q_eval, q_target)
                        writer.add_scalar('critic loss', critic_loss.numpy(), learn_steps)

                        # 梯度回传，优化网络参数
                        critic_optim.clear_grad()
                        critic_loss.backward()
                        critic_optim.step()

                        # 计算动作网络的损失函数
                        actor_loss = - critic(batch_state, actor(batch_state)).mean()
                        writer.add_scalar('actor loss', actor_loss.numpy(), learn_steps)

                        # 梯度回传，优化网络参数
                        actor_optim.clear_grad()
                        actor_loss.backward()
                        actor_optim.step()

                        # critic.train()

                        soft_update(actor_target, actor, tau)
                        soft_update(critic_target, critic, tau)

                episode_reward_list.append(episode_reward)
                writer.add_scalar('episode reward', episode_reward, epoch)
                paddle.save(actor.state_dict(), "linear_net.pdparams")
                makespan_temp = env.get_makespan()
                if makespan_temp < makespan:
                    makespan = makespan_temp
                if epoch % 10 == 0:
                    # print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
                    pbar.set_postfix({'episode': '%d' % (epochs / 10 * i + epoch + 1),
                                      'episode_reward_list': '%.3f' % np.mean(episode_reward_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(episode_reward_list)))
    mv_return = moving_average(episode_reward_list, 9)
    print('DDPG算法最好：{}'.format(makespan))
    print('DDPG算法最终：{}'.format(env.get_makespan()))
    csv_field['x'] = episodes_list
    csv_field['y' + str(num)] = mv_return
    makespan_field['best'].append(makespan)
    makespan_field['final'].append(env.get_makespan())

    print('DDPG算法最好：{}'.format(makespan))
    print('DDPG算法最终：{}'.format(env.get_makespan()))
    plt.title('DDPG')
    plt.plot(episodes_list, mv_return, linestyle='-')
    plt.show()
dataframe = pd.DataFrame(csv_field)
makespanFrame = pd.DataFrame(makespan_field)
# dataframe.to_csv("/opt/data/ddpg/ddpgreward20.csv", index=False, sep=',')
# makespanFrame.to_csv("/opt/data/ddpg/makespan20.csv", index=False, sep=',')
makespanFrame.to_csv("/opt/data/ddpg/makespanServer10.csv", index=False, sep=',')

