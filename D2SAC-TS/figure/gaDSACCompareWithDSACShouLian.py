import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('E:\\data\\sac\\sachasgareward（设置了fitness).csv')
df1 = pd.read_csv('E:\\data\\sac\\sacnogareward.csv')
# 提取数据列
x1 = df['x']
y1 = df['y0']

x2 = df1['x']
y2 = df1['y0']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 创建图表
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, color="red", linestyle='-', label='GA-DSAC')
plt.plot(x1, y2, color="blue", linestyle='-', label='DSAC')
font_size = 16
plt.legend()
plt.title('离散动作空间的算法收敛曲线', fontsize=font_size)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Episode Reward', fontsize=18)
plt.xticks(fontsize=font_size)  # 设置X轴刻度字体大小
plt.yticks(fontsize=font_size)
plt.grid(True)

# 显示图表
plt.show()
