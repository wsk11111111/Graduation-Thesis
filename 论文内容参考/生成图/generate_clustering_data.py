import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

np.random.seed(42)

def DS_1_1():
    X, y = make_blobs(n_samples=20000, centers=3, cluster_std=0.7, n_features=2, center_box=(-6, 6), random_state=42)
    return X, y

def DS_1_2():
    np.random.seed(42)
    X_list = []
    y_list = []

    n1 = 5000
    t1 = np.random.uniform(0, 1, n1)
    x1 = np.random.randn(n1) * 0.3
    y1 = (t1 - 0.5) * 16
    cluster1 = np.column_stack([x1, y1])
    X_list.append(cluster1)
    y_list.append(np.zeros(n1))

    n2 = 5000
    t2 = np.random.uniform(0, 1, n2)
    x2_base = np.where(t2 < 0.5, -2 + t2 * 4, 2 - (t2 - 0.5) * 4)
    y2_base = (t2 - 0.5) * 16
    x2 = x2_base + np.random.randn(n2) * 0.3
    y2 = y2_base + np.random.randn(n2) * 0.3
    cluster2 = np.column_stack([x2, y2])
    X_list.append(cluster2)
    y_list.append(np.ones(n2))

    n3 = 5000
    t3 = np.random.uniform(0, 1, n3)
    y3_base = (t3 - 0.5) * 16
    x3_base = np.where(y3_base < 0, -2 + (y3_base + 8) / 8 * 2, 2 - (y3_base - 0) / 8 * 2)
    x3 = x3_base + np.random.randn(n3) * 0.3
    y3 = y3_base + np.random.randn(n3) * 0.3
    cluster3 = np.column_stack([x3, y3])
    X_list.append(cluster3)
    y_list.append(np.ones(n3) * 2)

    n4 = 5000
    t4 = np.random.uniform(0, 1, n4)
    y4_base = (t4 - 0.5) * 16
    x4_left = -2 + (y4_base + 8) / 8 * 4 * 0.4
    x4_right = 2 - (y4_base + 8) / 8 * 4 * 0.4
    x4_base = np.where(y4_base < -4, x4_left, np.where(y4_base < 0, x4_right, x4_left))
    x4 = x4_base + np.random.randn(n4) * 0.3
    y4 = y4_base + np.random.randn(n4) * 0.3
    cluster4 = np.column_stack([x4, y4])
    X_list.append(cluster4)
    y_list.append(np.ones(n4) * 3)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.astype(int)

def DS_1_3():
    np.random.seed(42)
    X_list = []
    y_list = []

    center1 = [-6, 0]
    n1 = int(20000 * 0.35)
    r1 = np.sqrt(np.random.uniform(0, 1, n1)) * 3.5 + np.random.randn(n1) * 0.5
    theta1 = np.random.uniform(0, 2*np.pi, n1) + np.random.randn(n1) * 0.3
    cluster1 = np.column_stack([r1 * np.cos(theta1) + center1[0], r1 * np.sin(theta1) + center1[1]])
    X_list.append(cluster1)
    y_list.append(np.zeros(n1))

    center2 = [6, 0]
    n2 = int(20000 * 0.35)
    r2 = np.sqrt(np.random.uniform(0, 1, n2)) * 3.5 + np.random.randn(n2) * 0.5
    theta2 = np.random.uniform(0, 2*np.pi, n2) + np.random.randn(n2) * 0.3
    cluster2 = np.column_stack([r2 * np.cos(theta2) + center2[0], r2 * np.sin(theta2) + center2[1]])
    X_list.append(cluster2)
    y_list.append(np.ones(n2))

    center3 = [0, 7]
    n3 = 20000 - n1 - n2
    r3 = np.sqrt(np.random.uniform(0, 1, n3)) * 3.5 + np.random.randn(n3) * 0.5
    theta3 = np.random.uniform(0, 2*np.pi, n3) + np.random.randn(n3) * 0.3
    cluster3 = np.column_stack([r3 * np.cos(theta3) + center3[0], r3 * np.sin(theta3) + center3[1]])
    X_list.append(cluster3)
    y_list.append(np.ones(n3) * 2)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.astype(int)

def DS_1_4():
    np.random.seed(42)
    X_list = []
    y_list = []

    sparse1 = np.random.randn(5000, 2) * 2 + [-6, -6]
    X_list.append(sparse1)
    y_list.append(np.zeros(5000))

    compact1 = np.random.randn(3000, 2) * 0.4 + [6, -6]
    X_list.append(compact1)
    y_list.append(np.ones(3000))

    medium1 = np.random.randn(3000, 2) * 1.2 + [-6, 6]
    X_list.append(medium1)
    y_list.append(np.ones(3000) * 2)

    sparse2 = np.random.randn(2000, 2) * 2.5 + [6, 6]
    X_list.append(sparse2)
    y_list.append(np.ones(2000) * 3)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.astype(int)

def DS_1_5():
    np.random.seed(42)
    X_list = []
    y_list = []

    n_inner = 10000
    r_inner = np.random.uniform(1, 3, n_inner) + np.random.randn(n_inner) * 0.4
    theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
    inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    X_list.append(inner)
    y_list.append(np.zeros(n_inner))

    n_outer = 10000
    r_outer = np.random.uniform(5, 8, n_outer) + np.random.randn(n_outer) * 0.5
    theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
    outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    X_list.append(outer)
    y_list.append(np.ones(n_outer))

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.astype(int)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

datasets = [
    ('DS-1.1', DS_1_1),
    ('DS-1.2', DS_1_2),
    ('DS-1.3', DS_1_3),
    ('DS-1.4', DS_1_4),
    ('DS-1.5', DS_1_5)
]

for idx, (name, data_func) in enumerate(datasets):
    X, y = data_func()

    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(len(np.unique(y))):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=1, alpha=0.6)

    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

table_data = {
    'Dataset': ['DS-1.1', 'DS-1.2', 'DS-1.3', 'DS-1.4', 'DS-1.5'],
    'Samples': [20000, 20000, 20000, 13000, 20000],
    'Clusters': [3, 4, 3, 4, 2],
    'Shape': ['3 spherical\n(uniform density)', '4 curved strips\n(uniform density)', '3 clusters\n(non-uniform density)', '4 clusters\n(mixed density)', '2 concentric\nrings']
}

df = pd.DataFrame(table_data)

fig, ax = plt.subplots(figsize=(12, 2))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#E8E8E8'] * len(df.columns)
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for i in range(len(df.columns)):
    table[(0, i)].set_text_props(weight='bold')

plt.tight_layout()
plt.savefig('dataset_table.png', dpi=150, bbox_inches='tight', pad_inches=0.5)
plt.close()

print("All datasets and table generated and saved")