import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# only for test, to check the code, Only select part of samples
SAMPLE_FRAC = 1
SAMPLE_FRAC_TEST = 1

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

def _dirichlet_partition_data(y_data, n_clients, n_classes, alpha, min_size=1):
    """
    使用迪利克雷分布对CIFAR-10数据进行划分（与0919保持一致）
    """
    # 按类别收集数据索引
    class_indices = [np.where(y_data == i)[0] for i in range(n_classes)]
    
    # 初始化客户端数据索引
    client_data_indices = [[] for _ in range(n_clients)]
    
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
                remaining_clients = n_clients - client_id - 1
                remaining_min_samples = remaining_clients * min_samples_per_client
                available_samples = len(class_idx) - allocated_samples - remaining_min_samples
                num_samples = min(num_samples, available_samples)
                num_samples = max(num_samples, 0)  # 确保非负
            
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

def partition_data(dataset, datadir, partition, n_nets, alpha):
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    
    if partition == "hetero":
        # 使用迪利克雷分布划分训练集
        train_net_dataidx_map = _dirichlet_partition_data(
            y_train, n_nets, 10, alpha, min_size=10
        )
        
        # 使用相同方法划分测试集
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test, n_nets, 10, alpha, min_size=5
        )
        
        # 记录数据分布统计
        logging.info("CIFAR-10数据分布统计:")
        for client_id, data_indices in train_net_dataidx_map.items():
            if len(data_indices) > 0:
                client_labels = y_train[data_indices]
                class_counts = np.zeros(10)
                for label in client_labels:
                    class_counts[label] += 1
                logging.info(f"客户端 {client_id} - 训练样本数: {len(data_indices)}, "
                           f"测试样本数: {len(test_net_dataidx_map[client_id])}")
            
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)

# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)

def get_dataloader_CIFAR10(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = CIFAR10_truncated
    transform_train, transform_test = _data_transforms_cifar10()

    # 使用不同的索引创建训练集和测试集
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
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])

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
        train_dataidxs = train_net_dataidx_map[process_id - 1]
        test_dataidxs = test_net_dataidx_map[process_id - 1]
        local_data_num = len(train_dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_CIFAR10(data_dir, batch_size, batch_size,
                                                 train_dataidxs, test_dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num

def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    # 获取分割的数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # 计算总的训练和测试样本数
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    class_num = len(np.unique(y_train))
    
    # 获取全局数据加载器
    train_data_global, test_data_global = get_dataloader_CIFAR10(
        data_dir, batch_size, batch_size)
    
    # 为每个客户端创建本地数据加载器和记录数据量
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
        
        logging.info(f"Client {client_idx} - Training samples: {local_data_num}, "
                    f"Test samples: {len(test_dataidxs)}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num