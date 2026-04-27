import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

def save_clustering_plot(X, y, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

save_dir = r"C:\Mine\工作学习\项目\生成图\new"

print("Generating DS-1.1: 3 spherical clusters with uniform density...")
X1, y1 = make_blobs(n_samples=20000, centers=3, cluster_std=0.8,
                    center_box=(-10, 10), random_state=42)
save_clustering_plot(X1, y1, f"{save_dir}\\DS-1.1_spherical_clusters.png")

print("Generating DS-1.2: 4 letter/number shaped clusters...")
def generate_letter_clusters(n_samples, random_state=42):
    np.random.seed(random_state)
    X_list = []
    y_list = []

    def generate_stroke(points, n_pts_total, width):
        all_points = []
        segment_count = len(points) - 1
        n_pts_per_seg = n_pts_total // segment_count

        for i in range(segment_count):
            start, end = points[i], points[i+1]
            t = np.linspace(0, 1, n_pts_per_seg + 1)[:-1]
            x = start[0] + (end[0] - start[0]) * t + np.random.normal(0, width/3, n_pts_per_seg)
            y = start[1] + (end[1] - start[1]) * t + np.random.normal(0, width/3, n_pts_per_seg)
            all_points.extend(zip(x, y))

        if len(all_points) < n_pts_total:
            start, end = points[-2], points[-1]
            remaining = n_pts_total - len(all_points)
            t = np.linspace(0, 1, remaining + 1)[:-1]
            x = start[0] + (end[0] - start[0]) * t + np.random.normal(0, width/3, remaining)
            y = start[1] + (end[1] - start[1]) * t + np.random.normal(0, width/3, remaining)
            all_points.extend(zip(x, y))

        return np.array(all_points)

    pts_per_cluster = n_samples // 4
    width = 0.7

    letter_S = generate_stroke([(-6, 3), (-4, 3), (-4, 0), (-6, 0), (-6, -3), (-4, -3)], pts_per_cluster, width)
    X_list.append(letter_S)
    y_list.append(np.zeros(len(letter_S)))

    letter_K = generate_stroke([(-2, 3), (-2, -3), (-2, 1), (-0.5, 0), (-2, -1)], pts_per_cluster, width)
    X_list.append(letter_K)
    y_list.append(np.ones(len(letter_K)))

    number_3 = generate_stroke([(2, 3), (4, 3), (4, 0), (4, -3), (2, -3)], pts_per_cluster, width)
    X_list.append(number_3)
    y_list.append(np.full(len(number_3), 2))

    letter_M = generate_stroke([(6, 3), (6, -3), (8, 0), (10, 3), (10, -3)], pts_per_cluster, width)
    X_list.append(letter_M)
    y_list.append(np.full(len(letter_M), 3))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

X2, y2 = generate_letter_clusters(20000, random_state=42)
save_clustering_plot(X2, y2, f"{save_dir}\\DS-1.2_letter_clusters.png")

print("Generating DS-1.3: 3 clusters with non-uniform density (circular, separated)...")
def generate_nonuniform_density_clusters_separated(n_samples, random_state=42):
    np.random.seed(random_state)
    X_list = []
    y_list = []

    cluster_configs = [
        {'center': (-12, 0), 'sub_centers': [(-12, 0), (-10.5, 0.8), (-13.5, -0.5)], 'stds': [0.4, 0.6, 0.5], 'ratios': [0.5, 0.25, 0.25], 'n_samples': int(n_samples * 0.4)},
        {'center': (0, 0), 'sub_centers': [(0, 0), (2, 1), (-2, -0.8)], 'stds': [0.5, 0.7, 0.6], 'ratios': [0.4, 0.3, 0.3], 'n_samples': int(n_samples * 0.35)},
        {'center': (12, 0), 'sub_centers': [(12, 0), (13.5, 0.6), (10.5, -0.7)], 'stds': [0.6, 0.8, 0.5], 'ratios': [0.45, 0.3, 0.25], 'n_samples': int(n_samples * 0.25)}
    ]

    for config in cluster_configs:
        cluster_samples = []
        for sub_center, std, ratio in zip(config['sub_centers'], config['stds'], config['ratios']):
            n_pts = int(config['n_samples'] * ratio)
            samples = np.random.randn(n_pts, 2) * std + np.array(sub_center)
            cluster_samples.append(samples)
        X_cluster = np.vstack(cluster_samples)
        X_list.append(X_cluster)
        y_list.append(np.full(len(X_cluster), len(y_list)))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

X3, y3 = generate_nonuniform_density_clusters_separated(20000, random_state=42)
save_clustering_plot(X3, y3, f"{save_dir}\\DS-1.3_nonuniform_density.png")

print("Generating DS-1.4: 4 clusters with mixed density (separated)...")
def generate_mixed_density_clusters_separated(n_samples, random_state=42):
    np.random.seed(random_state)
    X_list = []
    y_list = []

    cluster_configs = [
        {'center': (-10, -8), 'std': 0.5, 'n_samples': int(n_samples * 0.15)},
        {'center': (10, -8), 'std': 0.6, 'n_samples': int(n_samples * 0.15)},
        {'center': (-10, 8), 'std': 3.0, 'n_samples': int(n_samples * 0.35)},
        {'center': (10, 8), 'std': 2.5, 'n_samples': int(n_samples * 0.35)}
    ]

    remaining = n_samples - sum(p['n_samples'] for p in cluster_configs)
    cluster_configs[2]['n_samples'] += remaining

    for i, params in enumerate(cluster_configs):
        samples = np.random.randn(params['n_samples'], 2) * params['std'] + np.array(params['center'])
        X_list.append(samples)
        y_list.append(np.full(params['n_samples'], i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

X4, y4 = generate_mixed_density_clusters_separated(13000, random_state=42)
save_clustering_plot(X4, y4, f"{save_dir}\\DS-1.4_mixed_density.png")

print("Generating DS-1.5: 2 concentric ring clusters with rough edges...")
def generate_concentric_rings_rough(n_samples, random_state=42):
    np.random.seed(random_state)
    X_list = []
    y_list = []

    ring_configs = [
        {'radius': 3, 'width': 0.8, 'n_samples': n_samples // 2, 'roughness': 0.3},
        {'radius': 7, 'width': 1.0, 'n_samples': n_samples // 2, 'roughness': 0.4}
    ]

    if n_samples % 2 == 1:
        ring_configs[0]['n_samples'] += 1

    for i, params in enumerate(ring_configs):
        theta = np.random.uniform(0, 2*np.pi, params['n_samples'])
        r = params['radius'] + np.random.uniform(-params['width']/2, params['width']/2, params['n_samples'])

        rough_noise = np.random.normal(0, params['roughness'], params['n_samples'])
        r = r + rough_noise

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        X_ring = np.column_stack([x, y])
        X_list.append(X_ring)
        y_list.append(np.full(params['n_samples'], i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

X5, y5 = generate_concentric_rings_rough(20000, random_state=42)
save_clustering_plot(X5, y5, f"{save_dir}\\DS-1.5_concentric_rings.png")

print("\nAll datasets generated and saved successfully!")