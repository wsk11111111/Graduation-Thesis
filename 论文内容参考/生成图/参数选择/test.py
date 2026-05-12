import matplotlib.pyplot as plt
import numpy as np

# --- 1. 基础设置 ---
# 设置中文字体，确保能显示中文（Windows常用 'SimHei'，Mac常用 'Arial Unicode MS'）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100  # 设置分辨率

# --- 2. 数据准备 ---
# X轴数据 (k值)
# x = list(range(1, 31))  # k从1到32
x_ticks = [0, 5, 10, 15, 20, 25,30,35,40]
x_labels = ['0', '5', '10', '15', '20', '25', '30', '35', '40']

# Y轴数据 (NMI值) - 根据原图目测估算的数值
# 你可以直接修改下面的列表来更新图表
data = {
    'DS-2.1': {
        'y': [0.46, 0.51, 0.52, 0.55, 0.55, 0.55, 0.55, 0.51, 0.52],
        'color': '#1f77b4'  # 蓝色
    },
    'DS-2.2': {
        'y': [0.63, 0.67, 0.71, 0.72, 0.72, 0.71, 0.73, 0.71, 0.69],
        'color': '#ff7f0e'  # 橙色
    },
    'DS-2.3': {
        'y': [0.34,0.52,0.55,0.60,0.61,0.62,0.62, 0.57, 0.56],
        'color': '#2ca02c'  # 绿色
    },
    'DS-2.4': {
        'y': [0.31, 0.35,0.39, 0.41, 0.41, 0.40, 0.39, 0.40, 0.41],
        'color': '#d62728'  # 红色
    },
    'DS-2.5': {
        'y': [0.69, 0.71, 0.72,0.74,0.74, 0.74, 0.75, 0.74, 0.75],
        'color': '#9467bd'  # 紫色
    }
}

# --- 3. 绘图 ---
plt.figure(figsize=(10, 6))

# 循环绘制每一条线
for label, ds in data.items():
    plt.plot(x_ticks, ds['y'],
             marker='o',            # 统一标记：圆圈
             linestyle='-',       # 统一线条：实线
             color=ds['color'],     # 颜色
             label=label,           # 图例标签
             markersize=4)          # 标记大小

# --- 4. 格式化图表 ---
plt.xlabel('迭代次数T', fontsize=12)
plt.ylabel('ARI', fontsize=12,rotation=0, labelpad=20)  # Y轴标签旋转0度并增加间距

# 设置X轴刻度 (每隔5个单位显示一个刻度)
plt.xticks(np.arange(0, 42, 5))

# 设置Y轴范围
plt.ylim(0.3, 0.8)

# 添加图例 (位置自动调整以防遮挡)
plt.legend(loc='upper right', frameon=True, fontsize=10)

# 添加网格 (可选，让图表更易读)
plt.grid(True, linestyle='--', alpha=0.6)

# 紧凑布局
plt.tight_layout()

# 显示图表
plt.show()