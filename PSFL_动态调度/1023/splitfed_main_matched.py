import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys
import random
import argparse
import logging

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from client_resource_allocation import (
    build_initial_client_profiles,
    device_tier_label,
    mutate_device_tiers,
    sample_resources_for_device_tier,
)
from model.resnet import get_resnet_num_blocks, get_tier_shared_block_counts


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="SplitFed: Split Federated Learning (Matched Resources)")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn", "fashion_mnist", "cinic10"],
        help="数据集名称",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--partition_method", type=str, default="hetero", help="数据划分方法")
    parser.add_argument("--partition_alpha", type=float, default=0.5, help="Dirichlet分布参数")

    parser.add_argument("--model", type=str, default="resnet56", choices=["resnet56", "resnet110"], help="模型架构")

    parser.add_argument("--client_number", type=int, default=10, help="客户端数量")
    parser.add_argument("--clients_per_round", type=int, default=0, help="每轮参与训练客户端数量(0表示全量)")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--rounds", type=int, default=100, help="联邦学习轮数")
    parser.add_argument("--local_epochs", type=int, default=1, help="本地训练轮数")

    parser.add_argument("--lr", type=float, default=0.005, help="学习率")
    parser.add_argument("--lr_factor", type=float, default=0.9, help="学习率衰减因子")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")

    parser.add_argument("--pretrain_epochs", type=int, default=10, help="预训练轮数")
    parser.add_argument("--pretrain_lr", type=float, default=0.01, help="预训练学习率")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()
    return args


class AutoFlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SplitFedClientModel(nn.Module):
    def __init__(self, input_channels=3, model_type="resnet56"):
        super(SplitFedClientModel, self).__init__()
        if model_type == "resnet56":
            num_blocks = [9, 9, 9]
        elif model_type == "resnet110":
            num_blocks = [18, 18, 18]
        else:
            num_blocks = [9, 9, 9]

        self.model_type = model_type
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 16, num_blocks[0])

        client_layer2_blocks = num_blocks[1] // 2
        self.layer2_client = self._make_layer(BasicBlock, 16, 32, client_layer2_blocks, stride=2)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2_client(x)
        return x


class SplitFedServerModel(nn.Module):
    def __init__(self, num_classes=10, model_type="resnet56"):
        super(SplitFedServerModel, self).__init__()
        if model_type == "resnet56":
            num_blocks = [9, 9, 9]
        elif model_type == "resnet110":
            num_blocks = [18, 18, 18]
        else:
            num_blocks = [9, 9, 9]

        self.model_type = model_type
        server_layer2_blocks = num_blocks[1] - num_blocks[1] // 2
        self.layer2_server = self._make_layer(BasicBlock, 32, 32, server_layer2_blocks)

        self.layer3 = self._make_layer(BasicBlock, 32, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer2_server(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _splitfed_client_compute_fraction(model_type: str) -> float:
    if model_type == "resnet110":
        l1, l2, l3 = 18, 18, 18
    else:
        l1, l2, l3 = 9, 9, 9
    client_layer2 = l2 // 2
    total_blocks = float(max(1, l1 + l2 + l3))
    return float((l1 + client_layer2) / total_blocks)


def _simulate_resource_delay(client, measured_train_time: float, model_type: str) -> float:
    try:
        compute_power = float(getattr(client, "compute_power", 1.0) or 1.0)
    except Exception:
        compute_power = 1.0
    try:
        network_speed = float(getattr(client, "network_speed", 50.0) or 50.0)
    except Exception:
        network_speed = 50.0

    compute_power = max(0.05, min(1.0, compute_power))
    network_speed = max(1.0, network_speed)
    measured_train_time = float(max(0.0, measured_train_time))

    tier = int(getattr(client, "model_tier", getattr(client, "tier", 4)) or 4)
    num_blocks = get_resnet_num_blocks(model_type)
    total_blocks = float(max(1, sum(int(x) for x in num_blocks)))
    shared = get_tier_shared_block_counts(model_type=model_type, tier=tier)
    shared_blocks = float(int(shared["layer1"]) + int(shared["layer2"]) + int(shared["layer3"]))

    if int(shared["layer3"]) > 0:
        activation_units = 64 * 8 * 8
    elif int(shared["layer2"]) > 0:
        activation_units = 32 * 16 * 16
    else:
        activation_units = 16 * 32 * 32

    client_compute_fraction = max(0.0, min(1.0, shared_blocks / total_blocks))
    comm_scale = float(activation_units) / float(32 * 16 * 16)

    compute_delay = measured_train_time * client_compute_fraction * max(0.0, (1.0 / compute_power) - 1.0) * 0.60
    comm_delay = measured_train_time * comm_scale * max(0.0, (50.0 / network_speed) - 1.0) * 0.10
    return float(max(0.0, compute_delay + comm_delay))


class SplitFedClient:
    def __init__(
        self,
        client_id,
        train_data,
        test_data,
        device="cuda",
        device_tier=None,
        compute_power=None,
        network_speed=None,
        storage_capacity=None,
        heterogeneity_score=None,
        model_tier=None,
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.model = None

        self.device_tier = device_tier
        self.compute_power = compute_power
        self.network_speed = network_speed
        self.storage_capacity = storage_capacity
        self.heterogeneity_score = heterogeneity_score
        self.model_tier = model_tier

    def set_model(self, model):
        self.model = model.to(self.device)

    def update_resources(self, device_tier=None, compute_power=None, network_speed=None, storage_capacity=None):
        if device_tier is not None:
            self.device_tier = device_tier
        if compute_power is not None:
            self.compute_power = compute_power
        if network_speed is not None:
            self.network_speed = network_speed
        if storage_capacity is not None:
            self.storage_capacity = storage_capacity

    def pretrain(self, epochs, lr):
        if self.model is None:
            raise ValueError("Client model not set")

        logging.info("Client %s starting pre-training...", self.client_id)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_loss = float("inf")
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                features = self.model(data)
                loss = torch.mean(torch.abs(features))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(1, batch_count)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0:
                logging.info("  Client %s Pre-train Epoch %s/%s, Loss: %.4f", self.client_id, epoch + 1, epochs, avg_loss)

        logging.info("Client %s pre-training completed, Best Loss: %.4f", self.client_id, best_loss)

    def train_step(self, server_model, optimizer, criterion, model_type: str):
        self.model.train()
        server_model.train()

        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for data, target in self.train_data:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            client_features = self.model(data)
            outputs = server_model(client_features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        accuracy = 100.0 * correct / max(1, total)

        measured_time = float(time.time() - start_time)
        simulated_delay = _simulate_resource_delay(self, measured_time, model_type=model_type)

        return avg_loss, accuracy, measured_time, simulated_delay

    def evaluate(self, server_model):
        self.model.eval()
        server_model.eval()

        correct = 0
        total = 0
        test_loss = 0.0
        batch_count = 0

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                client_features = self.model(data)
                outputs = server_model(client_features)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                batch_count += 1

        avg_loss = test_loss / max(1, batch_count)
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy, total


def load_dataset(args):
    logging.info("Loading dataset: %s", args.dataset)

    if args.dataset == "cifar10":
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    elif args.dataset == "svhn":
        data_loader = load_partition_data_svhn
    elif args.dataset == "fashion_mnist":
        data_loader = load_partition_data_fashion_mnist
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = "./data/cinic10/"
    else:
        data_loader = load_partition_data_cifar10

    train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = data_loader(
        args.dataset, args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size
    )

    logging.info("Total training samples: %s", train_data_num)
    logging.info("Total testing samples: %s", test_data_num)
    logging.info("Number of classes: %s", class_num)
    return train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def aggregate_models(client_models, client_data_sizes):
    total_data = sum(client_data_sizes.values())
    aggregated_state = {}
    first_client_id = list(client_models.keys())[0]
    param_keys = client_models[first_client_id].state_dict().keys()

    for key in param_keys:
        weighted_sum = None
        for client_id, model in client_models.items():
            weight = client_data_sizes[client_id] / max(1, total_data)
            param = model.state_dict()[key]
            if "num_batches_tracked" in key:
                if weighted_sum is None:
                    weighted_sum = param
                else:
                    weighted_sum = torch.max(weighted_sum, param)
            else:
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
        aggregated_state[key] = weighted_sum

    return aggregated_state


def main():
    args = parse_arguments()

    log_filename = f"{args.dataset}_{args.model}_splitfed_matched_alpha{str(args.partition_alpha).replace('.', 'p')}.txt"

    file_handler = AutoFlushFileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def flush_logs():
        for handler in logger.handlers:
            handler.flush()

    logging.info("=" * 80)
    logging.info("SplitFed: Split Federated Learning (Matched Resources)")
    logging.info("=" * 80)
    logging.info("Configuration:")
    for arg in vars(args):
        logging.info("  %s: %s", arg, getattr(args, arg))
    logging.info("=" * 80)
    flush_logs()

    set_seed(args.seed)
    logging.info("Random seed set to: %s", args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = load_dataset(args)
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    client_profiles = build_initial_client_profiles(
        client_number=args.client_number,
        train_data_local_dict=train_data_local_dict,
        n_classes=class_num,
        seed=args.seed,
    )
    client_resources = {
        cid: {
            "device_tier": profile.device_tier,
            "model_tier": profile.model_tier,
            "heterogeneity_score": profile.heterogeneity_score,
            "compute_power": profile.compute_power,
            "network_speed": profile.network_speed,
            "storage_capacity": profile.storage_capacity,
        }
        for cid, profile in client_profiles.items()
    }

    logging.info("为 %s 个客户端分配设备资源(5级)...", args.client_number)
    for cid in range(args.client_number):
        info = client_resources[cid]
        logging.info(
            "客户端 %s 资源: device_tier=%s(%s), heterogeneity=%.4f, compute_power=%.3f, network_speed=%s, storage_capacity=%s",
            cid,
            int(info["device_tier"]),
            device_tier_label(int(info["device_tier"])),
            float(info["heterogeneity_score"]),
            float(info["compute_power"]),
            int(info["network_speed"]),
            int(info["storage_capacity"]),
        )

    clients = {}
    for client_id in range(args.client_number):
        info = client_resources[client_id]
        client = SplitFedClient(
            client_id=client_id,
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            device=device,
            device_tier=int(info["device_tier"]),
            compute_power=float(info["compute_power"]),
            network_speed=int(info["network_speed"]),
            storage_capacity=int(info["storage_capacity"]),
            heterogeneity_score=float(info["heterogeneity_score"]),
            model_tier=int(info["model_tier"]),
        )
        clients[client_id] = client
        logging.info("Created Client %s, Training samples: %s", client_id, train_data_local_num_dict[client_id])

    client_models = {}
    for client_id in range(args.client_number):
        model = SplitFedClientModel(input_channels=input_channels, model_type=args.model)
        clients[client_id].set_model(model)
        client_models[client_id] = model

    server_model = SplitFedServerModel(num_classes=class_num, model_type=args.model).to(device)
    flush_logs()

    logging.info("\n" + "=" * 80)
    logging.info("Starting Pre-training Phase")
    logging.info("=" * 80)

    for client_id, client in clients.items():
        client.pretrain(epochs=args.pretrain_epochs, lr=args.pretrain_lr)
        flush_logs()

    logging.info("Pre-training Phase Completed\n")
    flush_logs()

    logging.info("=" * 80)
    logging.info("Starting Federated Learning Training")
    logging.info("=" * 80)

    best_accuracy = 0.0
    best_round = 0

    for round_idx in range(args.rounds):
        round_start = time.time()
        logging.info("\n%s", "=" * 80)
        logging.info("Round %s/%s", round_idx + 1, args.rounds)
        logging.info("%s", "=" * 80)

        clients_per_round = int(args.clients_per_round or 0)
        if clients_per_round <= 0 or clients_per_round > int(args.client_number):
            clients_per_round = int(args.client_number)
        selection_seed_base = 424242 + int(args.seed) * 1000
        round_rng = random.Random(selection_seed_base + round_idx)
        selected_client_ids = round_rng.sample(range(int(args.client_number)), k=clients_per_round)
        selected_client_ids.sort()
        logging.info("本轮参与训练客户端: %s", selected_client_ids)

        if (round_idx + 1) % 20 == 0:
            old_device_tiers = {cid: int(info["device_tier"]) for cid, info in client_resources.items()}
            mutation_seed = selection_seed_base + (round_idx + 1)
            new_device_tiers = mutate_device_tiers(old_device_tiers, fraction=0.30, seed=mutation_seed)

            changed = [
                cid for cid in range(args.client_number) if int(new_device_tiers[cid]) != int(old_device_tiers[cid])
            ]
            logging.info("每20轮触发动态设备变更: 选择 %s/%s 个客户端 (30%%)", len(changed), args.client_number)

            for client_id in changed:
                old_device = int(old_device_tiers[client_id])
                new_device = int(new_device_tiers[client_id])
                resample_seed = 100000000 + selection_seed_base * 10000 + (round_idx + 1) * 1000 + client_id
                rng = random.Random(resample_seed)
                compute_power, network_speed, storage_capacity = sample_resources_for_device_tier(new_device, rng=rng)

                client_resources[client_id]["device_tier"] = new_device
                client_resources[client_id]["compute_power"] = compute_power
                client_resources[client_id]["network_speed"] = network_speed
                client_resources[client_id]["storage_capacity"] = storage_capacity
                clients[client_id].update_resources(
                    device_tier=new_device,
                    compute_power=compute_power,
                    network_speed=network_speed,
                    storage_capacity=storage_capacity,
                )
                logging.info(
                    "客户端 %s 设备变更: device_tier %s(%s) -> %s(%s), heterogeneity=%.4f, compute_power=%.3f, network_speed=%s, storage_capacity=%s",
                    client_id,
                    old_device,
                    device_tier_label(old_device),
                    new_device,
                    device_tier_label(new_device),
                    float(client_resources[client_id]["heterogeneity_score"]),
                    float(compute_power),
                    network_speed,
                    storage_capacity,
                )

        current_lr = args.lr * (args.lr_factor ** (round_idx // 10))

        all_params = []
        for cid in selected_client_ids:
            all_params.extend(list(client_models[cid].parameters()))
        all_params.extend(list(server_model.parameters()))

        optimizer = torch.optim.Adam(all_params, lr=current_lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        train_accs = []

        per_client_times = []
        per_client_measured_times = []
        per_client_simulated_times = []
        training_wallclock_start = time.time()
        for client_id in selected_client_ids:
            client = clients[client_id]
            loss, acc, measured_time, simulated_delay = client.train_step(server_model, optimizer, criterion, model_type=args.model)
            train_losses.append(loss)
            train_accs.append(acc)
            per_client_times.append(measured_time + simulated_delay)
            per_client_measured_times.append(float(measured_time))
            per_client_simulated_times.append(float(measured_time + simulated_delay))

            logging.info(
                "  Client %s | device_tier=%s | Train Loss: %.4f, Acc: %.2f%% | time=%.3fs (raw=%.3fs, delay=%.3fs)",
                client_id,
                getattr(client, "device_tier", None),
                loss,
                acc,
                measured_time + simulated_delay,
                measured_time,
                simulated_delay,
            )
        training_wallclock = float(time.time() - training_wallclock_start)

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_train_acc = float(np.mean(train_accs)) if train_accs else 0.0

        selected_models = {cid: client_models[cid] for cid in selected_client_ids}
        selected_sizes = {cid: train_data_local_num_dict[cid] for cid in selected_client_ids}
        aggregation_start = time.time()
        aggregated_state = aggregate_models(selected_models, selected_sizes)
        for _, model in client_models.items():
            model.load_state_dict(aggregated_state)
        aggregation_time = float(time.time() - aggregation_start)

        test_losses = []
        test_accs = []
        test_samples = []

        for client_id, client in clients.items():
            loss, acc, samples = client.evaluate(server_model)
            test_losses.append(loss)
            test_accs.append(acc)
            test_samples.append(samples)

        total_samples = sum(test_samples)
        weighted_avg_acc = sum(acc * samples for acc, samples in zip(test_accs, test_samples)) / max(1, total_samples)
        avg_test_loss = float(np.mean(test_losses)) if test_losses else 0.0

        round_time = float(time.time() - round_start)
        train_time_real = float(sum(per_client_measured_times)) if per_client_measured_times else 0.0
        train_time_simulated = float(sum(per_client_simulated_times)) if per_client_simulated_times else 0.0

        logging.info("\nRound %s Statistics:", round_idx + 1)
        logging.info("  Average Train Loss: %.4f", avg_train_loss)
        logging.info("  Average Train Accuracy: %.2f%%", avg_train_acc)
        logging.info("  Average Test Loss: %.4f", avg_test_loss)
        logging.info("  Averaged Test Accuracy: %.2f%%", weighted_avg_acc)
        logging.info("  Learning Rate: %.6f", current_lr)
        logging.info(
            "  轮次总时间: %.2f秒, 训练(真实墙钟): %.2f秒, 训练(真实累计): %.2f秒, 训练(模拟累计): %.2f秒, 聚合: %.2f秒",
            float(round_time),
            float(training_wallclock),
            float(train_time_real),
            float(train_time_simulated),
            float(aggregation_time),
        )

        if weighted_avg_acc > best_accuracy:
            best_accuracy = float(weighted_avg_acc)
            best_round = round_idx + 1
            torch.save(
                {
                    "round": best_round,
                    "client_models": {cid: model.state_dict() for cid, model in client_models.items()},
                    "server_model": server_model.state_dict(),
                    "accuracy": best_accuracy,
                    "client_resources": client_resources,
                },
                f"{args.dataset}_{args.model}_splitfed_matched_alpha{str(args.partition_alpha).replace('.', 'p')}_best.pth",
            )
            logging.info("  *** New Best Accuracy: %.2f%% (Round %s) ***", best_accuracy, best_round)

        flush_logs()

    logging.info("\n" + "=" * 80)
    logging.info("Training Completed!")
    logging.info("Best Test Accuracy: %.2f%% (Round %s)", best_accuracy, best_round)
    logging.info("=" * 80)

    flush_logs()
    for handler in logger.handlers:
        handler.close()

    logging.info("\nLog file saved to: %s", log_filename)


if __name__ == "__main__":
    main()
