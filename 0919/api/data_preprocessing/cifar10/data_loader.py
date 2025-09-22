import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# only for test, to check the code, Only select part of samlpes

SAMPLE_FRAC = 1 # mean every 1000 select one sample to test
SAMPLE_FRAC_TEST = 1 # mean every 1000 select one sample to test


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data_dirichlet(dataset, datadir, partition_method, n_nets, alpha):
    """
    使用迪利克雷分布划分数据集，实现可控的数据异质性
    
    Args:
        dataset: 数据集名称
        datadir: 数据目录
        partition_method: 划分方法
        n_nets: 客户端数量
        alpha: 迪利克雷分布参数，控制异质性程度 (0.1-1.0)
               alpha越小异质性越强，alpha=1.0接近IID分布
    """
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    
    n_classes = 10  # CIFAR-10有10个类别
    n_samples = y_train.shape[0]
    
    # 创建数据索引映射
    train_net_dataidx_map = {}
    test_net_dataidx_map = {}
    
    if partition_method == "hetero":
        # 使用迪利克雷分布划分训练集
        train_net_dataidx_map = _dirichlet_partition_data(
            y_train, n_nets, n_classes, alpha, min_size=10
        )
        
        # 使用相同的分布比例划分测试集
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test, n_nets, n_classes, alpha, min_size=5,
            reference_distribution=_calculate_client_class_distributions(y_train, train_net_dataidx_map, n_classes)
        )
        
        # 记录数据分布统计
        _log_data_distribution_statistics(y_train, train_net_dataidx_map, n_classes, "训练集")
        _log_data_distribution_statistics(y_test, test_net_dataidx_map, n_classes, "测试集")
        
        # 生成数据分布可视化
        _generate_data_distribution_visualization(y_train, train_net_dataidx_map, n_classes)
    
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

def _dirichlet_partition_data(y_data, n_clients, n_classes, alpha, min_size=1, reference_distribution=None):
    """
    使用迪利克雷分布对数据进行划分
    
    Args:
        y_data: 标签数据
        n_clients: 客户端数量
        n_classes: 类别数量
        alpha: 迪利克雷分布参数
        min_size: 每个客户端最小样本数
        reference_distribution: 参考分布（用于保持测试集和训练集分布一致）
    """
    # 按类别收集数据索引
    class_indices = [np.where(y_data == i)[0] for i in range(n_classes)]
    
    # 初始化客户端数据索引
    client_data_indices = [[] for _ in range(n_clients)]
    
    if reference_distribution is not None:
        # 使用参考分布（用于测试集）
        for class_id in range(n_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            
            # 使用参考分布的比例
            proportions = reference_distribution[:, class_id]
            proportions = proportions / proportions.sum()  # 重新归一化
            
            # 分配样本
            start_idx = 0
            for client_id in range(n_clients):
                num_samples = int(proportions[client_id] * len(class_idx))
                end_idx = start_idx + num_samples
                
                if client_id == n_clients - 1:  # 最后一个客户端获得剩余所有样本
                    end_idx = len(class_idx)
                
                client_data_indices[client_id].extend(class_idx[start_idx:end_idx])
                start_idx = end_idx
    else:
        # 生成迪利克雷分布的分配比例
        for class_id in range(n_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            
            # 使用迪利克雷分布生成分配比例
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            
            # 确保每个客户端至少有min_size个样本
            min_samples_per_client = min(min_size, len(class_idx) // n_clients)
            allocated_samples = 0
            
            for client_id in range(n_clients):
                if client_id == n_clients - 1:
                    # 最后一个客户端获得剩余所有样本
                    num_samples = len(class_idx) - allocated_samples
                else:
                    num_samples = max(min_samples_per_client, 
                                    int(proportions[client_id] * len(class_idx)))
                    num_samples = min(num_samples, len(class_idx) - allocated_samples - (n_clients - client_id - 1) * min_samples_per_client)
                
                start_idx = allocated_samples
                end_idx = start_idx + num_samples
                
                client_data_indices[client_id].extend(class_idx[start_idx:end_idx])
                allocated_samples += num_samples
    
    # 打乱每个客户端的数据索引并转换为字典格式
    net_dataidx_map = {}
    for client_id in range(n_clients):
        np.random.shuffle(client_data_indices[client_id])
        net_dataidx_map[client_id] = client_data_indices[client_id]
    
    return net_dataidx_map

def _calculate_client_class_distributions(y_data, net_dataidx_map, n_classes):
    """计算客户端的类别分布矩阵"""
    n_clients = len(net_dataidx_map)
    distributions = np.zeros((n_clients, n_classes))
    
    for client_id, data_indices in net_dataidx_map.items():
        client_labels = y_data[data_indices]
        for class_id in range(n_classes):
            distributions[client_id, class_id] = np.sum(client_labels == class_id)
    
    return distributions

def _log_data_distribution_statistics(y_data, net_dataidx_map, n_classes, dataset_name):
    """记录数据分布统计信息"""
    logging.info(f"\n{dataset_name} 数据分布统计:")
    
    # 计算每个客户端的类别分布
    total_samples = 0
    class_totals = np.zeros(n_classes)
    
    for client_id, data_indices in net_dataidx_map.items():
        client_labels = y_data[data_indices]
        client_total = len(data_indices)
        total_samples += client_total
        
        class_counts = np.zeros(n_classes)
        for class_id in range(n_classes):
            count = np.sum(client_labels == class_id)
            class_counts[class_id] = count
            class_totals[class_id] += count
        
        # 计算基尼系数衡量数据不平衡程度
        if client_total > 0:
            proportions = class_counts / client_total
            proportions_sorted = np.sort(proportions)
            gini = 1 - 2 * np.sum((np.arange(n_classes) + 1) * proportions_sorted) / n_classes + 1
        else:
            gini = 0
        
        # 记录主要类别
        main_classes = np.where(proportions > 0.1)[0]
        
        logging.info(f"  客户端 {client_id}: {client_total} 样本, "
                    f"主要类别: {main_classes.tolist()}, "
                    f"基尼系数: {gini:.3f}")
    
    # 记录总体统计
    logging.info(f"  总样本数: {total_samples}")
    logging.info(f"  每类样本数: {class_totals.astype(int)}")

def _generate_data_distribution_visualization(y_train, train_net_dataidx_map, n_classes):
    """生成数据分布可视化"""
    try:
        from utils.data_visualization import DataDistributionVisualizer
        
        # 创建可视化器
        visualizer = DataDistributionVisualizer(save_dir="./visualizations")
        
        # 生成可视化图表
        distribution_matrix = visualizer.visualize_client_data_distribution(
            y_train, train_net_dataidx_map, n_classes, "CIFAR10"
        )
        
        logging.info("CIFAR-10数据分布可视化图表已生成")
        
        return distribution_matrix
        
    except ImportError:
        logging.warning("无法导入数据可视化工具，跳过图表生成")
        return None
    except Exception as e:
        logging.warning(f"生成数据分布可视化时出错: {str(e)}")
        return None

# 修改原有的partition_data函数以保持向后兼容性
def partition_data(dataset, datadir, partition_method, n_nets, alpha):
    """数据划分函数 - 使用迪利克雷分布实现"""
    return partition_data_dirichlet(dataset, datadir, partition_method, n_nets, alpha)


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = CIFAR10_truncated
    transform_train, transform_test = _data_transforms_cifar10()

    # Use different indices for training and test sets
    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=True, 
                     transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, 
                    transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, 
                              shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, 
                             shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num



def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    # Get partitioned data
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # Calculate total training and test samples
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    class_num = len(np.unique(y_train))
    
    # Get global data loaders
    train_data_global, test_data_global = get_dataloader_CIFAR10(
        data_dir, batch_size, batch_size)
    
    # Create local data loaders and record data amount for each client
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]
        
        # Record data amount for each client
        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        
        # Create local data loaders
        train_data_local, test_data_local = get_dataloader_CIFAR10(
            data_dir, batch_size, batch_size, 
            train_dataidxs, test_dataidxs)
            
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        logging.info(f"Client {client_idx} - Training samples: {local_data_num}, "
                    f"Test samples: {len(test_dataidxs)}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_partition_data_cifar10_with_clustering(dataset, data_dir, partition_method, partition_alpha, 
                                               client_number, batch_size, num_clusters=3, 
                                               clustering_method='cosine_similarity'):
    """加载并分区CIFAR-10数据集，并进行客户端聚类 - 主函数"""
    
    # 导入聚类管理器
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
    from utils.client_clustering import ClientClusterManager
    
    # 获取分区数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # 计算训练和测试样本总数
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    # 类别数量
    class_num = len(np.unique(y_train))
    
    # 获取全局数据加载器
    train_data_global, test_data_global = get_dataloader_CIFAR10(
        data_dir, batch_size, batch_size)
    
    # 创建本地数据加载器和记录每个客户端的数据量
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]
        
        # 记录每个客户端的数据量
        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        
        # 创建本地数据加载器
        train_data_local, test_data_local = get_dataloader_CIFAR10(
            data_dir, batch_size, batch_size, 
            train_dataidxs, test_dataidxs)
            
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        logging.info(f"客户端 {client_idx} - 训练样本数: {local_data_num}, "
                   f"测试样本数: {len(test_dataidxs)}")
    
    # 执行客户端聚类
    logging.info("开始CIFAR-10客户端聚类分析...")
    cluster_manager = ClientClusterManager(num_classes=class_num)
    
    # 计算客户端数据分布
    cluster_manager.calculate_client_distributions(
        train_data_local_dict, y_train, train_net_dataidx_map
    )
    
    # 执行聚类
    cluster_assignments = cluster_manager.cluster_clients(
        num_clusters=num_clusters, method=clustering_method
    )
    
    # 生成聚类可视化
    try:
        from utils.data_visualization import DataDistributionVisualizer
        visualizer = DataDistributionVisualizer(save_dir="./visualizations")
        
        # 计算分布矩阵
        distribution_matrix = np.zeros((client_number, class_num))
        for client_id in range(client_number):
            client_indices = train_net_dataidx_map[client_id]
            client_labels = y_train[client_indices]
            for class_id in range(class_num):
                count = np.sum(client_labels == class_id)
                if len(client_indices) > 0:
                    distribution_matrix[client_id, class_id] = count / len(client_indices)
        
        visualizer.visualize_clustering_results(distribution_matrix, cluster_assignments, "CIFAR10")
    except Exception as e:
        logging.warning(f"生成聚类可视化失败: {str(e)}")
    
    logging.info(f"CIFAR-10客户端聚类完成，聚类分配: {cluster_assignments}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, \
           cluster_manager