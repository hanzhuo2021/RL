

# 设计奖励函数（核心）
class Reward:
    def getReward(self, action):
        reward = []
        cpuInfo = (np.array(100 * len(self.execNodes)) - np.array(self.nodeCpuUsage)) * np.array(self.nodeCores)
        memInfo = (np.array(100 * len(self.execNodes)) - np.array(self.nodeMemUsage)) * np.array(self.nodeMem)
        # print("cpuInfo: ", cpuInfo)
        # print("memInfo: ", memInfo)
        for k in range(len(self.execNodes)):
            r = 0.0
            self.offlineDataCnt[k] += self.taskList[self.taskIdx[self.currTask]][0]
            dataCnt = np.array(self.nodeOnlineDataCnt) + np.array(self.offlineDataCnt)
            # print("dataCnt: ", dataCnt)
            for i in range(len(self.execNodes)):
                for j in range(i + 1, len(self.execNodes)):
                    reward_1 = sqrt(pow(dataCnt[i] / cpuInfo[i] - dataCnt[j] / cpuInfo[j], 2))
                    reward_2 = sqrt(pow(dataCnt[i] / memInfo[i] - dataCnt[j] / memInfo[j], 2))
                    r += reward_2 + reward_1
            self.offlineDataCnt[k] -= self.taskList[self.taskIdx[self.currTask]][0]
            # print("r: ", r)
            reward.append(1 / r)
        self.offlineDataCnt[action] += self.taskList[self.taskIdx[self.currTask]][0]
        # print("reward_cal: ", reward)
        # print(reward)
        if action == np.argmax(reward):
            return reward[action]
        else:
            return 0