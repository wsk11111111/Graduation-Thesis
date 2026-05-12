import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 方法 1: 全局设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示问题


# 数据
x = ['0.05', '0.01', '0.005', '0.001', '0.0005', '0.0001']
DS2_1 = [0.383, 0.536, 0.536, 0.591, 0.591, 0.591]  # DS2.1 数据（少一个值）
# DS2_2 = [0.504, 0.58, 0.627, 0.622, 0.647, 0.634]  # DS2.2 数据
DS2_2 = [0.547, 0.578, 0.579, 0.574, 0.647, 0.648]
DS2_3 = [0.562, 0.615, 0.668, 0.672, 0.730, 0.74]  # DS2.3 数据
DS2_4 = [0.449, 0.435, 0.427, 0.531, 0.585, 0.597]  # DS2.4 数据
# DS2_5 = [0.109, 0.316, 0.312, 0.337, 0.337, 0.337]  # DS2.5 数据
DS2_5 = [0.189, 0.333, 0.333, 0.333, 0.410, 0.418]

# 去掉 DS2.1 的最后一个值，使其与其他数据对齐
DS2_1 = DS2_1[:len(DS2_2)]

# 设置图形
plt.figure(figsize=(10, 6))

# 绘制每一条折线
plt.plot(x, DS2_1, label="DS1.1", marker='s', markersize=10, linestyle=':')  # 方形标记
plt.plot(x, DS2_2, label="DS1.2", marker='^', markersize=10, linestyle='-')  # 三角形标记
plt.plot(x, DS2_3, label="DS1.3", marker='D', markersize=10, linestyle='-.')  # 菱形标记
plt.plot(x, DS2_4, label="DS1.4", marker='x', markersize=10,linestyle='--')  # 十字形标记
plt.plot(x, DS2_5, label="DS1.5", marker='o', markersize=10, linestyle=':')  # 五角星标记


# 添加标题和标签
plt.xlabel("停止阈值",fontsize=18)
plt.ylabel("NMI",fontsize=18)
plt.ylim(0.1, 0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 自定义图例标记形状
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=20, label='DS1.1'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=20, label='DS1.2'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=20, label='DS1.3'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='purple', markersize=20, label='DS1.4'),
    Line2D([0], [0], marker='p', color='w', markerfacecolor='orange', markersize=20, label='DS1.5')
]



# 显示图例
plt.legend()
plt.ylim(0.1, 0.8)
# 显示网格
# plt.grid(True)

# 显示图形
plt.tight_layout()
plt.savefig('5.svg', format='svg')
plt.show()
