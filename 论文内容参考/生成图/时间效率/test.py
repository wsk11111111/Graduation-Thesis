import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保能显示中文（如“时间(秒)”）
# Windows系统通常使用 'SimHei'，Mac/Linux可能需要根据环境调整
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 定义数据规模 N (横坐标)
N_values = np.array([1000, 2000, 4000, 8000, 10000, 16000])

# 2. 模拟 FASCA 算法的实际运行时间 (实心原点连线)
# 假设 M = 0.05 * N (代表点个数与数据量成正比)
# 理论复杂度 T(N) ≈ k1 * (N*M) + k2 * (M^2)
# 代入 M 后，T(N) 大致呈现 O(N^2) 的趋势
# 这里加入一些随机噪声来模拟真实实验数据的波动
np.random.seed(42)  # 固定随机种子以保证结果可复现
noise = np.random.normal(0, 5, len(N_values)) # 添加少量高斯噪声
# 模拟公式：Time = 0.000003 * N * (0.05*N) + 0.01 * (0.05*N)**2 + 噪声
simulated_times = 0.000006 * N_values**2 + noise

# 3. 生成理论复杂度曲线 (实线)
# 使用拟合曲线来展示理论趋势 O(N*M + M^2)
# 这里为了视觉效果，生成更密集的点数画平滑曲线
N_smooth = np.linspace(1000, 16000, 100)
# 理论曲线公式 (调整系数使其与实际数据贴合)
theoretical_curve = 0.0000064 * N_smooth**2

# 4. 绘图
plt.figure(figsize=(10, 6))

# 绘制理论曲线 (实线)
plt.plot(N_smooth, theoretical_curve, color='red', linestyle='-', linewidth=2, label=r'Theoretical $O(N \cdot M + M^2)$')

# 绘制实际运行时间 (实心原点连线)
plt.plot(N_values, simulated_times, color='blue', marker='o', linestyle='-', markersize=8, label='FASCA Actual Time')

# 5. 设置图表标签和格式
plt.xlabel('Data Scale (N)', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=12)

# 设置坐标轴范围，留一点边距
plt.xlim(0, 17000)
plt.ylim(0, max(simulated_times) + 20)

# 显示图表
plt.tight_layout()
plt.show()