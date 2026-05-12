import numpy as np
from sklearn.neighbors import NearestNeighbors


def row_normalize(matrix):
    # 计算每行的和
    row_sums = matrix.sum(axis=1, keepdims=True)
    # 防止除以0的情况（避免出现行和为0的情况）
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums

    return normalized_matrix


def print_non_zero_elements(matrix):
    # 遍历矩阵的每一行
    for i, row in enumerate(matrix):
        # 获取当前行中非零的元素
        non_zero_elements = row[row != 0]

        # 打印当前行中非零的元素
        if non_zero_elements.size > 0:
            print(f"Row {i}: {non_zero_elements}")
def compute_matrix(data3, kmeans_selected_points, labels, k):
    n_points = len(kmeans_selected_points)  # 聚类中心的数量
    P1_matrix = np.zeros((n_points, n_points)) #+ np.eye(n_points) # 初始化P1
    P2_matrix = np.zeros((n_points, n_points)) #+ np.eye(n_points) # 初始化P2
    P3_matrix = np.zeros((n_points, n_points)) #+ np.eye(n_points) # 初始化P3
    # 使用 NearestNeighbors 获取每个聚类中心的 k 个近邻（只在聚类中心之间找）
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(kmeans_selected_points)  # +1 是因为包括自身
    distances, indices = nbrs.kneighbors(kmeans_selected_points)

    # 计算每个点的局部密度
    local_density = np.zeros(n_points)

    for i in range(n_points):
        a = kmeans_selected_points[i]
        a_label = i
        a_cluster = data3[labels == a_label]
        distances_to_center = np.linalg.norm(a_cluster - a, axis=1)
        # # 计算局部密度：簇内每个点到中心的距离之和 / 簇内点数
        # # local_density[i] = np.sum(distances_to_center) / len(a_cluster)
        # local_density[i] = len(a_cluster) / np.sum(distances_to_center)
        if (np.sum(distances_to_center) == 0):
            local_density[i] = 0
        else:
            local_density[i] = len(a_cluster) / np.sum(distances_to_center)

     # 计算每个点的平均邻居距离 t
    avg_distances = np.zeros(n_points)  # 存储每个点的平均距离t
    for i in range(n_points):
        # 获取当前点的k个近邻（排除自身）
        neighbors_distances = distances[i, 1:]  # 排除自己
        # 计算当前点的平均距离t
        avg_distances[i] = np.mean(neighbors_distances)

    for i in range(n_points):
        neighbors_indices = indices[i, 1:]  # 排除自己

        a = kmeans_selected_points[i]  # 当前聚类中心a
        a_local_density = local_density[i]  # 当前点a的局部密度
        a_label = i  # 聚类中心a所属的簇
        a_cluster = data3[labels == a_label]  # a所在的簇
        t_a = avg_distances[i]  # 当前点a的平均距离t


        for j in range(k):
            b_index = neighbors_indices[j]
            b = kmeans_selected_points[b_index]  # 点b（邻居聚类中心）
            b_local_density = local_density[b_index]  # 点b的局部密度
            b_label = labels[b_index]  # 聚类中心b所属的簇
            b_cluster = data3[labels == b_label]  # b所在的簇
            t_b = avg_distances[b_index]  # 点b的平均距离t

            # 计算P1：a和b之间的距离
            dist_a_to_b = np.linalg.norm(a - b)

            x = np.sum((np.linalg.norm(a_cluster - b, axis=1) <= dist_a_to_b))
            y = np.sum(np.linalg.norm(b_cluster - a, axis=1) <= dist_a_to_b)
            P1 = (x + y) / (len(a_cluster) + len(b_cluster))
            if(a_local_density * b_local_density == 0):
                P2 = 0
            else:
                P2 = min(a_local_density, b_local_density) / max(a_local_density, b_local_density)
            P3 = np.exp(-dist_a_to_b / (t_a + t_b))
            # 更新P1和P2矩阵
            P1_matrix[i, b_index] = P1
            P2_matrix[i, b_index] = P2
            P3_matrix[i, b_index] = P3

    combined_matrix = P1_matrix * P2_matrix * P3_matrix
    combined_matrix = row_normalize(combined_matrix)

    return combined_matrix
