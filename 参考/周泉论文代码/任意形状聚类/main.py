import pandas as pd
import pasco

def load_dataset(choice):
    data1 = None
    true_labels = None
    sample_rate = None
    cluster_num = None
    class_num = None
    contamination = 0.2

    if choice == 1:
        # 第一组数据
        df = pd.read_csv('../PASCO_DS21/wdbc.csv')
        data1 = df.iloc[1:, :-1].to_numpy()
        true_labels = df.iloc[1:, -1]
        cluster_num = 2
        class_num = 3
        sample_rate = 30
    elif choice == 2:
        # 第二组数据
        df = pd.read_csv('../PASCO_DS22/segmentation.data', header=None)
        true_labels = df.iloc[:, 0]
        data1 = df.iloc[:, 1:].to_numpy()
        # sample_rate = round(65)
        sample_rate = 0.07
        cluster_num = 7
        class_num = 2
        contamination = 0.15
    elif choice == 3:
        # 第三组数据
        df = pd.read_csv('../PASCO_DS23/sat.trn', delimiter=' ')
        data1 = df.iloc[:, :-1].to_numpy()
        true_labels = df.iloc[:, -1]
        # sample_rate = round(136)
        sample_rate = 0.03
        cluster_num = 6
        class_num = 4
        contamination = 0.15
    elif choice == 4:
        # 第四组数据
        df = pd.read_excel('../PASCO_DS24/Dry_Bean_Dataset.xlsx')
        # sample_rate = round(259)
        sample_rate = 0.06
        cluster_num = 7
        class_num = 5
        data1 = df.iloc[:, :-1].to_numpy()
        true_labels = df.iloc[:, -1]
    elif choice == 5:
        features_file = "../PASCO_DS25/pendigits_sta4_train.csv"
        labels_file = "../PASCO_DS25/pendigits_label_train.csv"
        data1 = pd.read_csv(features_file, header=None).to_numpy()  # 无表头，直接转为 NumPy 数组
        true_labels = pd.read_csv(labels_file, header=None).iloc[:, 0].to_numpy()
        class_num = 8
        cluster_num = 10
        # sample_rate = round(181)
        sample_rate = 0.04
    elif choice == 6:
        file_path = "../PASCO_DS26/letter-recognition.data"
        df = pd.read_csv(file_path, header=None)
        data1 = df.iloc[:, 1:].to_numpy()
        true_labels = df.iloc[:, 0].to_numpy()
        class_num = 4
        cluster_num = 26
        # sample_rate = round(1212)
        sample_rate = 0.09

    # 返回选择的数据和相关参数
    return data1, true_labels, sample_rate, cluster_num, class_num, contamination


# 数据集编号列表
dataset_ids = [2,3,4,5,6]

# 遍历每个数据集，执行实验并打印NMI
for dataset_id in dataset_ids:
    # 加载数据集
    data11, true_labels1, _, cluster_num1, _, contamination1 = load_dataset(dataset_id)

    # 设置实验参数
    sample_rate1 = 2
    class_num1 = 6

    # 进行PASCO实验，计算NMI
    nmi = pasco.pasco1(data11, true_labels1, cluster_num1, class_num1, sample_rate1, contamination1)
    nmi = round(nmi, 3)

    # 打印该数据集的NMI结果
    print(f"Dataset {dataset_id} NMI: {nmi}")
