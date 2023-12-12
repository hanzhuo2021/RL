import sys
import time

import numpy as np
import paddle
from matplotlib import pyplot as plt
from tqdm import tqdm
from visualdl import LogWriter

from ActorCriticModel import ActorCriticModel
from DE import DE
from Memory import Memory
from SAC import SAC
from SACAgent import SACAgent
# from SACDiscreteTest.env.myEnv import Environment
from ilabEnv.env import Environment
import sys
max_float = sys.float_info.max
from SACDiscreteTest.ga.ga import GA
from ilabEnv import task, node


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, action_dim, sac_agent, epochs, memory, warmup_steps, batch_size, is_use_DE = False):
    episode_reward_list = []
    max_episode_reward = sys.float_info.min
    learn_steps = 0
    population = []
    makespan = max_float
    for i in range(10):
        final_makespan_list = [max_float, max_float, max_float, max_float]
        with tqdm(total=int(epochs / 10), desc='Iteration %d' % i) as pbar:
            for epoch in range(int(epochs / 10)):
                # 状态初始化
                state = env.reset()
                # 智能体与环境交互一个回合的回合总奖励
                episode_reward = 0
                # 回合开始
                for time_step in range(10000):
                    learn_steps += 1
                    action = sac_agent.sample(state)
                    next_state, reward, done = env.step(action)

                    if done:
                        break
                    
                    episode_reward += reward
                    memory.append(state, action, reward, next_state)
                    state = next_state

                    # 收集到足够的经验后进行网络的更新
                    if memory.size() >= warmup_steps:
                        # 梯度更新
                        batch_state, batch_action, batch_reward, batch_next_state = memory.sample(batch_size)
                        critic_loss, actor_loss = sac_agent.learn(batch_state, batch_action, batch_reward,
                                                                  batch_next_state)
                        writer.add_scalar('critic loss', critic_loss.numpy(), learn_steps)
                        writer.add_scalar('actor loss', actor_loss.numpy(), learn_steps)
                if is_use_DE:
                    population.append(sac_agent.alg.get_actor())
                    if len(population) == 10:
                        batch_state, batch_action, batch_reward, batch_next_state = memory.sample(batch_size)
                        de = DE(population, batch_state, len(population))
                        best_policy = de.evolution()
                        with paddle.no_grad():
                            sac_agent.alg.hard_update_target(best_policy)
                        population = []

                if max_episode_reward < episode_reward:
                    max_episode_reward = episode_reward
                    sac_agent.alg.save()
                makespan_temp = env.get_makespan()
                if makespan_temp < makespan:
                    makespan = makespan_temp
                    # print(makespan)

                episode_reward_list.append(episode_reward)

                writer.add_scalar('episode reward', episode_reward, len(episode_reward_list))

                if (epoch + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (epochs / 10 * i + epoch + 1),
                                      'episode_reward_list': '%.3f' % np.mean(episode_reward_list[-10:])})
                pbar.update(1)

    return episode_reward_list, makespan


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epochs = 2000
    # epochs = 2500

    # 初始化超参数
    WARMUP_STEPS = 5000
    MEMORY_SIZE = int(1e6)
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.001
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA = 0.1

    # 定义环境、实例化模型
    # env = Environment()
    # state_dim = len(env.observation_space)
    # action_dim = len(env.action_space)
    state_dim = 44
    action_dim = 1
    # 初始化 模型，算法，智能体以及经验池
    model = ActorCriticModel(state_dim, action_dim)
    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=ALPHA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    sac_agent = SACAgent(algorithm)
    memory = Memory(max_size=MEMORY_SIZE, state_dim=state_dim, action_dim=action_dim)

    # 开始训练
    train_begin_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = './' + train_begin_time + '/log'
    model_dir = './' + train_begin_time + '/model'
    writer = LogWriter(logdir=log_dir)
    print('visualdl --logdir ' + log_dir + ' --port 8080')
    ga = GA(len(task.get_task_list()), task.get_NODE_LIST())
    ga.run()
    individuals = ga.get_top_n_individuals(1)
    fitness = individuals[0].cal_fitness()
    # 不使用遗传算法，不使用DE算法
    env1 = Environment()
    print("不使用遗传算法，不使用DE算法")
    episode_reward_list1, makespan1 = train_off_policy_agent(env1, action_dim, sac_agent, epochs, memory, WARMUP_STEPS, BATCH_SIZE, False)
    print(makespan1)

    # 使用遗传算法，不使用DE算法
    env2 = Environment()
    env2.set_fitness(fitness)
    print("使用遗传算法，不使用DE算法")
    episode_reward_list2, makespan2 = train_off_policy_agent(env2, action_dim, sac_agent, epochs, memory, WARMUP_STEPS, BATCH_SIZE, False)
    print(makespan2)

    # 使用遗传算法，使用DE算法
    # env3 = Environment()
    # env2.set_fitness(fitness)
    # print("使用遗传算法，使用DE算法")
    # episode_reward_list3, makespan3 = train_off_policy_agent(env3, action_dim, sac_agent, epochs, memory, WARMUP_STEPS, BATCH_SIZE, True)
    # print(makespan3)

    # 不使用遗传算法，使用DE算法
    # env4 = Environment()
    # print("不使用遗传算法，使用DE算法")
    # episode_reward_list4, makespan4 = train_off_policy_agent(env4, action_dim, sac_agent, epochs, memory, WARMUP_STEPS, BATCH_SIZE, True)
    # print(makespan4)

    # 可视化训练结果
    episodes_list = list(range(len(episode_reward_list2)))
    print('不使用遗传算法，不使用DE算法：{}'.format(env1.get_makespan()))
    print('使用遗传算法，不使用DE算法：{}'.format(env2.get_makespan()))
    # print('使用遗传算法，使用DE算法：{}'.format(env3.get_makespan()))
    # print('不使用遗传算法，使用DE算法：{}'.format(env4.get_makespan()))
    # print("最优的完成时间：")
    # for idx, makespan in enumerate(final_makespan_list):
    #     print(idx)
    #     print(":")
    #     print(makespan)
    mv_return1 = moving_average(episode_reward_list1, 9)
    mv_return2 = moving_average(episode_reward_list2, 9)
    # mv_return3 = moving_average(episode_reward_list3, 9)
    # mv_return4 = moving_average(episode_reward_list4, 9)

    fig, axs = plt.subplots(2, 2, figsize=(15, 18))
    axs[0, 0].plot(episodes_list, mv_return1, color='blue', label="noga no de")
    axs[0, 0].set_title('noga no de')
    axs[0, 0].legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    axs[0, 1].plot(episodes_list, mv_return2, color='blue', label="has ga no de")
    axs[0, 1].set_title('has ga no de')
    axs[0, 1].legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    # axs[1, 0].plot(episodes_list, mv_return3, color='blue', label="has ga has de")
    # axs[1, 0].set_title('has ga has de')
    # axs[1, 0].legend()
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    #
    # axs[1, 1].plot(episodes_list, mv_return4, color='blue', label="no ga has de")
    # axs[1, 1].set_title('no ga has de')
    # axs[1, 1].legend()
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')

    axs[1, 0].plot(episodes_list, mv_return1, c='green', label="NO GA, NO DE")
    axs[1, 0].plot(episodes_list, mv_return2, c='red', label="HAS GA, NO DE")
    # axs[1, 0].plot(episodes_list, mv_return3, c='blue', label="HAS GA, HAS DE")
    # axs[1, 0].plot(episodes_list, mv_return4, c='yellow', label="HAS GA, HAS DE")
    axs[1, 0].legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    # 隐藏第六个子图（不使用的）
    axs[1, 1].axis('off')

    # 自动调整子图布局
    plt.tight_layout()

    # 显示图表
    plt.show()

