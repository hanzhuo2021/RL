import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
# 读取CSV文件
task_num = [20, 30, 40, 50]
label_list = ['GA-DSAC', 'DDPG', 'GA', 'PPO', 'RR']
# gasac_best_makespan = []
# gasac_final_makespan = []
# ddpg_best_makespan = []
# ddpg_final_makespan = []
# ga_best_makespan = []
# ga_final_makespan = []
# ppo_best_makespan = []
# ppo_final_makespan = []
# RR_best_makespan = []
# RR_final_makespan = []

final_makespan = {}
best_makespan = {}

for num in task_num:
    gaSac = pd.read_csv('E:\\serverData\\data\\sac\\sachasgamakespan' + str(num) + '.csv')
    ddpg = pd.read_csv('E:\\serverData\\data\\ddpg\\makespan' + str(num) + '.csv')
    ga = pd.read_csv('E:\\serverData\\data\\GA\\makespan' + str(num) + '.csv')
    ppo = pd.read_csv('E:\\serverData\\data\\ppo\\makespan' + str(num) + '.csv')
    RR = pd.read_csv('E:\\serverData\\data\\RR\\makespan' + str(num) + '.csv')

    final_list = [gaSac.iloc[0, 1], ddpg.iloc[0, 1], ga.iloc[0, 1], ppo.iloc[0, 1], RR.iloc[0, 1]]
    final_makespan[str(num)] = final_list

    best_list = [gaSac.iloc[0, 0], ddpg.iloc[0, 0], ga.iloc[0, 0], ppo.iloc[0, 0], RR.iloc[0, 0]]
    best_makespan[str(num)] = best_list

final_makespan["Model"] = label_list
best_makespan["Model"] = label_list
    # gasac_best_makespan.append(gaSac.iloc[0, 0])
    # gasac_final_makespan.append(gaSac.iloc[0, 1])
    #
    # ddpg_best_makespan.append(ddpg.iloc[0, 0])
    # ddpg_final_makespan.append(ddpg.iloc[0, 1])
    #
    # ga_best_makespan.append(ga.iloc[0, 0])
    # ga_final_makespan.append(ga.iloc[0, 1])
    #
    # ppo_best_makespan.append(ppo.iloc[0, 0])
    # ppo_final_makespan.append(ppo.iloc[0, 1])
    #
    # RR_best_makespan.append(ppo.iloc[0, 0])
    # RR_final_makespan.append(ppo.iloc[0, 1])

data = pd.DataFrame(final_makespan)
# data = pd.DataFrame(best_makespan)

# 将数据转换成长格式
df_long = data.melt(id_vars='Model', var_name='Variable', value_name='Value')
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
sn.set(style="whitegrid", font_scale=1.2)
fontsize = 18
# 使用seaborn绘制多组柱状图
plt.figure(figsize=(12, 8))
ax = sn.barplot(x='Variable', y='Value', hue='Model', data=df_long, palette=custom_palette)
plt.xlabel('the number of tasks', fontsize=fontsize)
plt.ylabel('makespan', fontsize=fontsize)
# plt.title("best makespan", fontsize=fontsize)
plt.title("final makespan", fontsize=fontsize)
plt.grid(linestyle="--", alpha=0.3)

# 在每个柱子上显示数字
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.show()