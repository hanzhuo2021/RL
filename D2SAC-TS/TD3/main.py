from utils import str2bool,evaluate_policy, Reward_adapter
from datetime import datetime
from TD3 import TD3_agent
# import gymnasium as gym
import numpy as np
import os, shutil
import argparse
import torch
from ilabEnv.envYiLai import Environment
import sys
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from SACDiscreteTest.ga.ga import GA
from ilabEnv import task, node
max_float = sys.float_info.max

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.15, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    # ga = GA(len(task.get_task_list()), task.get_NODE_LIST())
    # ga.run()
    # individuals = ga.get_top_n_individuals(1)
    # fitness = individuals[0].cal_fitness()
    # Build Env
    env = Environment()
    # env.set_fitness(fitness)
    # eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = 44
    opt.action_dim = 1
    opt.max_action = float(1)   #remark: action space【-max,max】
    # opt.max_e_steps = env._max_episode_steps

    # Seed Everything
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    if not os.path.exists('model'): os.mkdir('model')
    agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        csv_field = {}
        makespan_field = {"best": [], "final": []}
        # 运行十次
        for num in range(1):
            episode_reward_list = []
            max_episode_reward = sys.float_info.min
            population = []
            makespan = max_float
            epochs = 2500
            warmup_steps = 5000
            for i in range(10):
                learn_steps = 0
                with tqdm(total=int(epochs / 10), desc='Iteration %d' % i) as pbar:
                    for epoch in range(int(epochs / 10)):
                        # 状态初始化
                        state = env.reset()
                        # 智能体与环境交互一个回合的回合总奖励
                        episode_reward = 0
                        # 回合开始
                        for time_step in range(10000):
                            learn_steps += 1
                            action = agent.select_action(state, False)
                            next_state, reward, done = env.step(action)

                            if done:
                                break

                            episode_reward += reward
                            # agent.memory.push(state, action, reward, next_state, done)
                            agent.replay_buffer.add(state, action, reward, next_state, done)
                            state = next_state

                            # 收集到足够的经验后进行网络的更新
                            if learn_steps >= warmup_steps:
                                # 梯度更新
                                agent.train()

                        # if max_episode_reward < episode_reward:
                        #     max_episode_reward = episode_reward
                        #     agent.alg.save()
                        makespan_temp = env.get_makespan()
                        if makespan_temp < makespan:
                            makespan = makespan_temp
                            # print(makespan)

                        episode_reward_list.append(episode_reward)

                        # writer.add_scalar('episode reward', episode_reward, len(episode_reward_list))

                        if (epoch + 1) % 10 == 0:
                            pbar.set_postfix({'episode': '%d' % (epochs / 10 * i + epoch + 1),
                                              'episode_reward_list': '%.3f' % np.mean(episode_reward_list[-10:])})
                        pbar.update(1)
            episodes_list = list(range(len(episode_reward_list)))
            mv_return = moving_average(episode_reward_list, 9)
            print('TD3算法最好：{}'.format(makespan))
            print('TD3算法最终：{}'.format(env.get_makespan()))
            csv_field['x'] = episodes_list
            csv_field['y' + str(num)] = mv_return
            makespan_field['best'].append(makespan)
            makespan_field['final'].append(env.get_makespan())
            plt.title('TD3')
            plt.plot(episodes_list, mv_return, linestyle='-')
            plt.show()
        dataframe = pd.DataFrame(csv_field)
        makespanFrame = pd.DataFrame(makespan_field)
        dataframe.to_csv("/opt/data/td3/reward20.csv", index=False, sep=',')
        makespanFrame.to_csv("/opt/data/td3/makespan20.csv", index=False, sep=',')
    # episodes_list = list(range(len(episode_reward_list)))
    # mv_return1 = moving_average(episode_reward_list, 9)
    # print('TD3算法最好：{}'.format(makespan))
    # print('TD3算法最终：{}'.format(env.get_makespan()))
    # plt.title('TD3')
    # plt.plot(episodes_list, mv_return1, linestyle='-')
    # plt.show()

if __name__ == '__main__':
    main()




