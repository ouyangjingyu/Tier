import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

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

def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar10_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def _calculate_client_class_distributions(y_data, net_dataidx_map, n_classes):
    n_clients = len(net_dataidx_map)
    distributions = np.zeros((n_clients, n_classes))
    for client_id, data_indices in net_dataidx_map.items():
        client_labels = y_data[data_indices]
        for class_id in range(n_classes):
            distributions[client_id, class_id] = np.sum(client_labels == class_id)
    return distributions


def _log_data_distribution_statistics(y_data, net_dataidx_map, n_classes, dataset_name):
    logging.info(f"\n{dataset_name} 数据分布统计:")

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

        if client_total > 0:
            proportions = class_counts / client_total
            proportions_sorted = np.sort(proportions)
            n = len(proportions_sorted)
            cumsum = np.cumsum(proportions_sorted)
            gini = (n + 1 - 2 * np.sum(cumsum)) / n if n > 0 else 0
        else:
            proportions = np.zeros(n_classes)
            gini = 0

        main_classes = np.where(proportions > 0.02)[0]
        logging.info(
            f"  客户端 {client_id}: {client_total} 样本, "
            f"主要类别(>2%): {main_classes.tolist()}, "
            f"基尼系数: {gini:.3f}"
        )

    logging.info(f"  总样本数: {total_samples}")
    logging.info(f"  每类样本数: {class_totals.astype(int).tolist()}")


def _generate_data_distribution_visualization(y_train, train_net_dataidx_map, n_classes):
    try:
        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        distributions = _calculate_client_class_distributions(
            y_train, train_net_dataidx_map, n_classes
        )
        proportions = distributions / (distributions.sum(axis=1, keepdims=True) + 1e-12)

        save_dir = "./visualizations"
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "cifar100_client_class_distribution_train.png")

        fig, ax = plt.subplots(
            figsize=(max(10, n_classes / 6), max(4, len(train_net_dataidx_map) / 2)),
            dpi=180,
        )
        im = ax.imshow(proportions, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xlabel("Class")
        ax.set_ylabel("Client")
        ax.set_title("CIFAR100 Train Client-Class Distribution")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"CIFAR100 数据分布可视化图表已生成: {out_path}")
        return proportions
    except ImportError as e:
        logging.warning(f"无法导入matplotlib，跳过图表生成: {str(e)}")
        return None
    except Exception as e:
        logging.warning(f"生成数据可视化时出错: {str(e)}")
        return None


def _dirichlet_partition_data(y_data, n_clients, n_classes, alpha, min_size=1, reference_distribution=None):
    class_indices = [np.where(y_data == i)[0] for i in range(n_classes)]
    client_data_indices = [[] for _ in range(n_clients)]

    if reference_distribution is not None:
        for class_id in range(n_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            proportions = reference_distribution[:, class_id]
            proportions = proportions / (proportions.sum() + 1e-8)
            expected = proportions * len(class_idx)
            counts = np.floor(expected).astype(int)
            remainder = int(len(class_idx) - counts.sum())
            if remainder > 0:
                frac = expected - counts
                order = np.argsort(frac)[::-1]
                counts[order[:remainder]] += 1

            start_idx = 0
            for client_id in range(n_clients):
                end_idx = start_idx + int(counts[client_id])
                client_data_indices[client_id].extend(class_idx[start_idx:end_idx])
                start_idx = end_idx
    else:
        for class_id in range(n_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))

            min_samples_per_client = min(min_size, len(class_idx) // n_clients)
            allocated_samples = 0

            for client_id in range(n_clients):
                if client_id == n_clients - 1:
                    num_samples = len(class_idx) - allocated_samples
                else:
                    num_samples = max(
                        min_samples_per_client, int(proportions[client_id] * len(class_idx))
                    )
                    remaining_clients = n_clients - client_id - 1
                    remaining_min_samples = remaining_clients * min_samples_per_client
                    available_samples = len(class_idx) - allocated_samples - remaining_min_samples
                    num_samples = min(num_samples, available_samples)
                    num_samples = max(num_samples, 0)

                start_idx = allocated_samples
                end_idx = start_idx + num_samples
                client_data_indices[client_id].extend(class_idx[start_idx:end_idx])
                allocated_samples += num_samples

    net_dataidx_map = {}
    for client_id in range(n_clients):
        np.random.shuffle(client_data_indices[client_id])
        net_dataidx_map[client_id] = client_data_indices[client_id]
    return net_dataidx_map


def partition_data_dirichlet(dataset, datadir, partition_method, n_nets, alpha):
    X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    n_classes = 100

    train_net_dataidx_map = {}
    test_net_dataidx_map = {}

    if partition_method == "hetero":
        train_net_dataidx_map = _dirichlet_partition_data(
            y_train, n_nets, n_classes, alpha, min_size=10
        )
        reference = _calculate_client_class_distributions(y_train, train_net_dataidx_map, n_classes)
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test, n_nets, n_classes, alpha, min_size=5, reference_distribution=reference
        )

        _log_data_distribution_statistics(y_train, train_net_dataidx_map, n_classes, "CIFAR100训练集")
        _log_data_distribution_statistics(y_test, test_net_dataidx_map, n_classes, "CIFAR100测试集")
        _generate_data_distribution_visualization(y_train, train_net_dataidx_map, n_classes)

    elif partition_method == "homo":
        idxs = np.random.permutation(y_train.shape[0])
        batch_idxs = np.array_split(idxs, n_nets)
        train_net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_nets)}

        reference = _calculate_client_class_distributions(y_train, train_net_dataidx_map, n_classes)
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test, n_nets, n_classes, alpha=1.0, min_size=1, reference_distribution=reference
        )
    else:
        raise NotImplementedError

    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map


def partition_data(dataset, datadir, partition, n_nets, alpha):
    if partition == "hetero-fix":
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        n_classes = 100
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt'
        train_net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)
        reference = _calculate_client_class_distributions(y_train, train_net_dataidx_map, n_classes)
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test, n_nets, n_classes, alpha=1.0, min_size=1, reference_distribution=reference
        )
        _log_data_distribution_statistics(y_train, train_net_dataidx_map, n_classes, "CIFAR100训练集(hetero-fix)")
        _log_data_distribution_statistics(y_test, test_net_dataidx_map, n_classes, "CIFAR100测试集(hetero-fix)")
        _generate_data_distribution_visualization(y_train, train_net_dataidx_map, n_classes)
        return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

    return partition_data_dirichlet(dataset, datadir, partition, n_nets, alpha)


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs, None)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR100(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cifar100(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )
    class_num = len(np.unique(y_train))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    batch_size_test = 100 # make all test on similar batch size
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size_test)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
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
        train_data_local, test_data_local = get_dataloader_CIFAR100(
            data_dir, batch_size, batch_size_test, train_dataidxs, test_dataidxs
        )
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None

    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar100(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )
    class_num = len(np.unique(y_train))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    
    batch_size_test = 100 # make all test on similar batch size

    train_data_global, test_data_global = get_dataloader_CIFAR100(data_dir, batch_size, batch_size_test)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]
        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_CIFAR100(
            data_dir, batch_size, batch_size_test, train_dataidxs, test_dataidxs
        )
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
