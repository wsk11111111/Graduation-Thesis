import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_files = ['ls3', 'd6', 'cth3']

for name in data_files:
    data = np.loadtxt(f'{name}.txt')
    labels = np.loadtxt(f'{name}_cl.txt').astype(int)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    cmap = plt.colormaps.get_cmap('tab10').resampled(n_clusters)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax1.scatter(data[mask, 0], data[mask, 1], c=[cmap(i)], label=f'Label {lbl}', s=10, alpha=0.7)
    ax1.set_title(f'{name} - Original Labels')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    fig1.savefig(f'{name}_original.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=300, random_state=42)
    pred_labels = kmeans.fit_predict(data)

    pred_unique = np.unique(pred_labels)
    cmap2 = plt.colormaps.get_cmap('tab10').resampled(len(pred_unique))

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(pred_unique):
        mask = pred_labels == lbl
        ax2.scatter(data[mask, 0], data[mask, 1], c=[cmap2(i)], label=f'Cluster {lbl}', s=10, alpha=0.7)
    ax2.set_title(f'{name} - KMeans Clustering (k={n_clusters}, iter=300)')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.legend()
    fig2.savefig(f'{name}_kmeans.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f'{name}: original labels={n_clusters}, saved {name}_original.png and {name}_kmeans.png')

print('All done!')
