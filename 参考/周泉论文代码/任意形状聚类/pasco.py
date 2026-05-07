import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import time
import moveR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def remove_outliers_LOF(data, contamination=0.20):
    lof = LocalOutlierFactor(n_neighbors=10, contamination = contamination)# contamination表示离群点的比例
    y_pred = lof.fit_predict(data)
    normal_indices = np.where(y_pred == 1)[0]
    outlier_indices = np.where(y_pred == -1)[0]
    filtered_data = data[y_pred == 1]
    outliers_data = data[y_pred == -1]
    return filtered_data, outliers_data, normal_indices, outlier_indices

def noirmalized(data):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized


def nor01(data):
    # 初始化Scaler
    scaler = MinMaxScaler()

    # 进行0-1缩放
    data1_scaled = scaler.fit_transform(data)
    return data1_scaled

def stand01(data1):
    scaler = StandardScaler()

    # 对 data1 进行标准化
    data1_scaled = scaler.fit_transform(data1)
    return data1_scaled

def sample(data, p_ratio):
    num = int(len(data)*p_ratio)
    non_outliers, _ ,_ ,_= remove_outliers_LOF(data)
    N,D = non_outliers.shape
    selected_points = np.zeros((num,D))
    distances = np.full(N,np.inf)
    center_point = np.mean(non_outliers, axis=0)
    distances_to_center = np.linalg.norm(non_outliers - center_point, axis=1)
    P1_index = np.argmin(distances_to_center)
    selected_points[0] = non_outliers[P1_index]
    distances = np.minimum(distances, np.linalg.norm(non_outliers - non_outliers[P1_index], axis=1))

    if p_ratio < 1:
        for i in range(1, num):
            next_index = np.argmax(distances)
            selected_points[i] = non_outliers[next_index]
            distances = np.minimum(distances, np.linalg.norm(non_outliers - non_outliers[next_index], axis=1))
    elif p_ratio > 3:
        num = p_ratio
        for i in range(1, num):
            next_index = np.argmax(distances)
            selected_points[i] = non_outliers[next_index]
            distances = np.minimum(distances, np.linalg.norm(non_outliers - non_outliers[next_index], axis=1))
    else:
        previous_distances = []
        max_distances_history = []  # 用于记录所有的 max_distance
        for i in range(1, num):
            next_index = np.argmax(distances)
            selected_points[i] = non_outliers[next_index]
            max_distance = np.max(distances)
            max_distances_history.append(max_distance)  # 记录当前的最大距离

            # 记录之前的最小距离
            previous_distances.append(max_distance)
            if len(previous_distances) > 5:
                previous_distances.pop(0)

            # 如果记录的最小距离超过5个，计算其平均值
            if len(previous_distances) == 5:
                avg_min_distance = np.mean(previous_distances)
                if max_distance >= 0.9999 * avg_min_distance:
                    break  # 满足结束条件则退出采样

            distances = np.minimum(distances, np.linalg.norm(non_outliers - non_outliers[next_index], axis=1))
        # windowSize = 10
        # previous_distances = []  # 用于记录所有的最大距离
        # max_distances_history = []  # 用于记录所有的 max_distance
        # stop_count = 0  # 用于统计满足条件的连续次数
        #
        # for i in range(1, num):
        #     # 选择距离最大的一点
        #     next_index = np.argmax(distances)
        #     selected_points[i] = non_outliers[next_index]
        #
        #     # 计算并记录当前的最大距离
        #     max_distance = np.max(distances)
        #     max_distances_history.append(max_distance)
        #
        #     # 记录当前最大距离到 previous_distances
        #     previous_distances.append(max_distance)
        #
        #     # 如果 previous_distances 的长度大于 windowSize，删除最早的最大距离
        #     if len(previous_distances) > windowSize:
        #         previous_distances.pop(0)
        #
        #     # 当 previous_distances 中有 windowSize 个最大距离时，检查是否满足停止条件
        #     if len(previous_distances) == windowSize:
        #         # 比较当前的最大距离和 windowSize 次前的最大距离的 99%
        #         distance_to_compare = previous_distances[0]  # 第一个采样点的最大距离（即第一个记录的最大距离）
        #
        #         # 判断当前最大距离与 distance_to_compare 是否满足条件（99%的变化）
        #         if max_distance >= 0.99 * distance_to_compare:
        #             stop_count += 1  # 如果满足条件，计数器加 1
        #         else:
        #             stop_count = 0  # 如果不满足条件，重置计数器
        #
        #         # 如果连续五次满足条件，停止采样
        #         if stop_count >= 5:
        #             break  # 如果连续五次满足条件，则退出采样
        #
        #     # 更新距离
        #     distances = np.minimum(distances, np.linalg.norm(non_outliers - non_outliers[next_index], axis=1))
    selected_points = selected_points[~np.all(selected_points == 0, axis=1)]
    print(f"采样点数{len(selected_points)}")
    return selected_points

def kmeans_sample(data,selected_points):
    k = selected_points.shape[0]
    kmeans = KMeans(n_clusters=k, init=selected_points, n_init=1).fit(data)
    k_means_selected_points = kmeans.cluster_centers_
    labels = kmeans.labels_
    closest_points = []

    # 对每个簇进行操作
    for i in range(k):
        # 获取该簇内的所有点
        cluster_points = data[labels == i]

        if len(cluster_points) == 0:
            closest_points.append(k_means_selected_points[i])
            continue

        # 计算这些点到簇中心的距离
        distances = np.linalg.norm(cluster_points - k_means_selected_points[i], axis=1)

        # 找到距离簇中心最近的点
        closest_point_idx = np.argmin(distances)
        closest_point = cluster_points[closest_point_idx]
        closest_points.append(closest_point)

    # 确保 closest_points 和 labels 之间的对应关系
    # 在这里，closest_points_labels 已经确保了 closest_points 与对应簇的标签一致
    return k_means_selected_points, labels, np.array(closest_points)


##需要修改
def refresh_reprepresentive_points(representive_points, attract, k_num):
    nbrs = NearestNeighbors(n_neighbors=k_num+1, algorithm='ball_tree').fit(representive_points)
    distances, indices = nbrs.kneighbors(representive_points)
    n_points = len(representive_points)
    influence_matrix = np.zeros((n_points, n_points))

    for i in range(n_points):
        neighbors_distances = distances[i, 1:]
        neighbors_indices = indices[i, 1:]

        total_inverse_distance = np.sum(1.0 / neighbors_distances)

        for j, neighbor_index in enumerate(neighbors_indices):
            influence = attract * (1.0 / neighbors_distances[j]) / total_inverse_distance
            influence_matrix[i, neighbor_index] = influence

        influence_matrix[i, i] = 1 - attract

    return influence_matrix

def merge_closest_pair(points, labels):
    n = len(points)
    if n <= 1:
        return points, labels

    min_dist = float('inf')
    closest_pair = (0, 1)

    points_np = np.array(points)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_np[i] - points_np[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j)

    idx1, idx2 = closest_pair
    merged_point = tuple((np.array(points[idx1]) + np.array(points[idx2])) / 2)
    merged_label = merge_labels(labels[idx1], labels[idx2])

    new_points = [points[k] for k in range(n) if k != idx1 and k != idx2]
    new_points.append(merged_point)

    new_labels = [labels[k] for k in range(n) if k != idx1 and k != idx2]
    new_labels.append(merged_label)

    return new_points, new_labels

def merge_labels(l1, l2):
    # Merge two labels into a set of unique labels and return as a tuple
    if isinstance(l1, tuple):
        l1 = set(l1)
    else:
        l1 = {l1}

    if isinstance(l2, tuple):
        l2 = set(l2)
    else:
        l2 = {l2}

    return tuple(l1.union(l2))

def merge(points, labels, num_clusters):
    Z = linkage(points, method='single')

    labels = fcluster(Z, num_clusters, criterion='maxclust')

    return labels

def updated_points2_label(points1, points2, labels):
    kdtree = KDTree(points1)
    _, indices1 = kdtree.query(points2)
    updated_points_2_labels = []
    for i in range(len(indices1)):
        updated_points_2_labels.append(labels[indices1[i]])
    return  updated_points_2_labels

def update_labels(labels, final_labels):
    # 创建一个映射字典，用于记录final_labels中每个元素对应的新标签
    label_mapping = {}
    next_label = 0  # 下一个可用的新标签

    for i, a in enumerate(final_labels):
        if a not in label_mapping:
            label_mapping[a] = next_label
            next_label += 1

        # 将labels中所有值为i的元素设置为a对应的新标签
        for idx in range(len(labels)):
            if labels[idx] == i:
                labels[idx] = label_mapping[a]

    # 统计labels中有多少不同的元素
    unique_labels = set(labels)

    # 将labels中的元素换成从0开始的顺序标签
    label_index_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_index_mapping[label] for label in labels]

    return labels

def assign_outliersdata_to_clusters(outliers_data, points):
    distances = cdist(outliers_data, points)
    closest_centers = np.argmin(distances, axis=1)
    return closest_centers

def calculate_accuracy(confusion_matrix):
    # 转化为 numpy 数组
    matrix = confusion_matrix.values
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-matrix)
    # 计算最佳匹配下的正确预测样本数
    correct_predictions = matrix[row_ind, col_ind].sum()
    # 计算准确率
    accuracy = correct_predictions / matrix.sum()
    return accuracy, row_ind, col_ind

def pasco1(data1, true_labels, cluster_num, class_num, sample_rate, contamination):
    # data2 = noirmalized(data1)  # 正则化
    # data2 = nor01(data1)
    # data2 = stand01(data1)
    data2 = noirmalized(stand01(data1))
    data3, outliers_data, data3_indices, outliers_data_indices = remove_outliers_LOF(data2, contamination)  # 去除离散点
    selected_points = sample(data3, sample_rate)  # picture1 采样
    print(len(selected_points))

    start_time = time.time()

    kmeans_selected_points, labels, new_kmeans_selected_points = kmeans_sample(data3, selected_points)  # 采样点聚类后调整聚类中心
    outliers_data_labels = assign_outliersdata_to_clusters(outliers_data, kmeans_selected_points)
    all_points_labels = np.copy(labels)
    for i, idx in enumerate(outliers_data_indices):
        all_points_labels = np.insert(all_points_labels, idx, outliers_data_labels[i])
    end_time_kmeans = time.time()
    # print(f"kmeans用时 {end_time_kmeans - start_time}seconds")

    # influence_matrix = pasco.refresh_reprepresentive_points(kmeans_selected_points, attract, class_num)
    influence_matrix = moveR.compute_matrix(data3, new_kmeans_selected_points, labels, class_num)
    updated_points = kmeans_selected_points
    for i in range(21):
        updated_points = np.dot(influence_matrix, updated_points)
    end_time_attract = time.time()
    # print(f"吸引力矩阵用时{end_time_attract - end_time_kmeans}seconds")

    updated_points_1, updated_points_0, updated_points_1_indices, updated_points_0_indices = remove_outliers_LOF(updated_points, 0.20)
    init_labels = list(range(len(updated_points_1)))
    updated_points_1_labels = merge(updated_points_1, init_labels, cluster_num)
    end_1 = time.time()
    # print(f"updated_points1_labels {end_1 - end_time_attract}seconds")
    updated_points_0_labels = updated_points2_label(updated_points_1, updated_points_0, updated_points_1_labels)
    end_2 = time.time()
    # print(f"updated_points2_labels {end_2 - end_1}seconds")
    final_labels = np.full(len(updated_points), -1)
    final_labels[updated_points_1_indices] = updated_points_1_labels
    final_labels[updated_points_0_indices] = updated_points_0_labels
    labels = update_labels(all_points_labels, final_labels)
    end_time = time.time()
    # print(f"最后合并用时{end_time - end_time_attract}seconds")
    print(f"总共用时{end_time - start_time}seconds")
    best_ari = adjusted_rand_score(true_labels, labels)
    best_ami = adjusted_mutual_info_score(true_labels, labels)
    # print(f"Adjusted Rand Index (ARI): {best_ari}")
    # print(f"Adjusted Rand Index (AMI): {best_ami}")
    # 计算NMI
    nmi = normalized_mutual_info_score(true_labels, labels)
    # print(f"Normalized Mutual Information (NMI): {nmi}")
    # 计算混淆矩阵
    confusion_matrix = pd.crosstab(pd.Series(true_labels, name='Actual'), pd.Series(labels, name='Predicted'))
    accuracy, row_ind, col_ind = calculate_accuracy(confusion_matrix)
    # print(f"Accuracy: {accuracy}")
    # # 打印最佳匹配
    # print("Best matching:")
    # for r, c in zip(row_ind, col_ind):
    #     print(f"Row {r} -> Column {c}")
    return nmi



