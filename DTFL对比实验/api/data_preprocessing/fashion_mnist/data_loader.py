import logging
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .datasets import FashionMNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _data_transforms_fashion_mnist():
    FASHION_MNIST_MEAN = [0.2860]  # Fashion-MNIST单通道均值
    FASHION_MNIST_STD = [0.3530]   # Fashion-MNIST单通道标准差

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整到32x32以适应ResNet
        transforms.Grayscale(num_output_channels=3),  # 转换为3通道
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2860, 0.2860, 0.2860], [0.3530, 0.3530, 0.3530]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.2860, 0.2860, 0.2860], [0.3530, 0.3530, 0.3530]),
    ])

    return train_transform, test_transform

def load_fashion_mnist_data(datadir):
    train_transform, test_transform = _data_transforms_fashion_mnist()

    fashion_mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    fashion_mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = fashion_mnist_train_ds.data, fashion_mnist_train_ds.target
    X_test, y_test = fashion_mnist_test_ds.data, fashion_mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition, n_nets, alpha):
    X_train, y_train, X_test, y_test = load_fashion_mnist_data(datadir)
    
    if partition == "hetero":
        # 为训练集和测试集分别创建索引映射
        train_net_dataidx_map = {}
        test_net_dataidx_map = {}
        
        # 分割训练集
        min_size = 0
        K = 10  # Fashion-MNIST有10个类别
        N = y_train.shape[0]
        
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                
        # 同样方式分割测试集
        test_idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(idx_k, proportions))]

        # 保存训练和测试索引映射
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            train_net_dataidx_map[j] = idx_batch[j]
            np.random.shuffle(test_idx_batch[j])
            test_net_dataidx_map[j] = test_idx_batch[j]
            
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

def get_dataloader_fashion_mnist(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = FashionMNIST_truncated
    transform_train, transform_test = _data_transforms_fashion_mnist()

    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=True, 
                     transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, 
                    transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, 
                              shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, 
                             shuffle=False, drop_last=False)

    return train_dl, test_dl

def load_partition_data_fashion_mnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    # 获取分割的数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # 计算总的训练和测试样本数
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    class_num = len(np.unique(y_train))
    
    # 获取全局数据加载器
    train_data_global, test_data_global = get_dataloader_fashion_mnist(
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
        train_data_local, test_data_local = get_dataloader_fashion_mnist(
            data_dir, batch_size, batch_size, 
            train_dataidxs, test_dataidxs)
            
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        logging.info(f"Client {client_idx} - Training samples: {local_data_num}, "
                    f"Test samples: {len(test_dataidxs)}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num