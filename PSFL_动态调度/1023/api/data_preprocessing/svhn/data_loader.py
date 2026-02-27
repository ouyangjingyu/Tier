import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import SVHN_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_svhn():
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_transform.transforms.append(Cutout(16))

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def load_svhn_data(datadir):
    train_transform, test_transform = _data_transforms_svhn()
    train_ds = SVHN_truncated(datadir, train=True, download=True, transform=train_transform)
    test_ds = SVHN_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = train_ds.data, train_ds.target
    X_test, y_test = test_ds.data, test_ds.target
    return (X_train, y_train, X_test, y_test)


def _calculate_client_class_distributions(y_data, net_dataidx_map, n_classes):
    n_clients = len(net_dataidx_map)
    distributions = np.zeros((n_clients, n_classes))
    for client_id, data_indices in net_dataidx_map.items():
        client_labels = y_data[data_indices]
        for class_id in range(n_classes):
            distributions[client_id, class_id] = np.sum(client_labels == class_id)
    return distributions


def _dirichlet_partition_data(
    y_data,
    n_clients,
    n_classes,
    alpha,
    min_size=1,
    reference_distribution=None,
):
    class_indices = [np.where(y_data == i)[0] for i in range(n_classes)]
    client_data_indices = [[] for _ in range(n_clients)]

    if reference_distribution is not None:
        for class_id in range(n_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)

            proportions = reference_distribution[:, class_id]
            if proportions.sum() <= 0:
                proportions = np.ones_like(proportions)
            proportions = proportions / proportions.sum()

            start_idx = 0
            for client_id in range(n_clients):
                num_samples = int(proportions[client_id] * len(class_idx))
                end_idx = start_idx + num_samples
                if client_id == n_clients - 1:
                    end_idx = len(class_idx)
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
                    num_samples = min(
                        num_samples,
                        len(class_idx)
                        - allocated_samples
                        - (n_clients - client_id - 1) * min_samples_per_client,
                    )

                start_idx = allocated_samples
                end_idx = start_idx + num_samples
                client_data_indices[client_id].extend(class_idx[start_idx:end_idx])
                allocated_samples += num_samples

    net_dataidx_map = {}
    for client_id in range(n_clients):
        np.random.shuffle(client_data_indices[client_id])
        net_dataidx_map[client_id] = client_data_indices[client_id]
    return net_dataidx_map


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
            gini = 1 - 2 * np.sum((np.arange(n_classes) + 1) * proportions_sorted) / n_classes + 1
        else:
            gini = 0

        main_classes = np.where((class_counts / max(1, client_total)) > 0.1)[0]
        logging.info(
            f"  客户端 {client_id}: {client_total} 样本, 主要类别: {main_classes.tolist()}, 基尼系数: {gini:.3f}"
        )

    logging.info(f"  总样本数: {total_samples}")
    logging.info(f"  每类样本数: {class_totals.astype(int)}")


def partition_data_dirichlet(dataset, datadir, partition_method, n_nets, alpha):
    X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    n_classes = 10

    train_net_dataidx_map = {}
    test_net_dataidx_map = {}

    if partition_method == "hetero":
        train_net_dataidx_map = _dirichlet_partition_data(
            y_train, n_nets, n_classes, alpha, min_size=10
        )
        test_net_dataidx_map = _dirichlet_partition_data(
            y_test,
            n_nets,
            n_classes,
            alpha,
            min_size=5,
            reference_distribution=_calculate_client_class_distributions(
                y_train, train_net_dataidx_map, n_classes
            ),
        )
        _log_data_distribution_statistics(y_train, train_net_dataidx_map, n_classes, "SVHN训练集")
        _log_data_distribution_statistics(y_test, test_net_dataidx_map, n_classes, "SVHN测试集")
    elif str(partition_method).lower() in {"homo", "iid"}:
        train_perm = np.random.permutation(np.arange(y_train.shape[0], dtype=np.int64))
        test_perm = np.random.permutation(np.arange(y_test.shape[0], dtype=np.int64))
        train_splits = np.array_split(train_perm, int(n_nets))
        test_splits = np.array_split(test_perm, int(n_nets))
        train_net_dataidx_map = {i: train_splits[i].astype(np.int64).tolist() for i in range(int(n_nets))}
        test_net_dataidx_map = {i: test_splits[i].astype(np.int64).tolist() for i in range(int(n_nets))}
        _log_data_distribution_statistics(y_train, train_net_dataidx_map, n_classes, "SVHN训练集")
        _log_data_distribution_statistics(y_test, test_net_dataidx_map, n_classes, "SVHN测试集")
    else:
        raise NotImplementedError

    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map


def partition_data(dataset, datadir, partition_method, n_nets, alpha):
    return partition_data_dirichlet(dataset, datadir, partition_method, n_nets, alpha)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_SVHN(datadir, train_bs, test_bs, dataidxs, None)


def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_SVHN(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_SVHN(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = SVHN_truncated
    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl


def get_dataloader_test_SVHN(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = SVHN_truncated
    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl


def load_partition_data_svhn(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )

    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    class_num = len(np.unique(y_train))

    train_data_global, test_data_global = get_dataloader_SVHN(data_dir, batch_size, batch_size)

    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]

        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num

        train_data_local, test_data_local = get_dataloader_SVHN(
            data_dir, batch_size, batch_size, train_dataidxs, test_dataidxs
        )

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

        logging.info(
            f"Client {client_idx} - Training samples: {local_data_num}, Test samples: {len(test_dataidxs)}"
        )

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

