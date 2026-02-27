import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import random
import argparse
import logging
from datetime import datetime

# 设置环境变量
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ.setdefault('OMP_NUM_THREADS', '1')

# 添加路径（假设与您的项目结构一致）
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入数据加载模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
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
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SplitFed: Split Federated Learning')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'fashion_mnist', 'cinic10'],
                       help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='数据划分方法')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='Dirichlet分布参数')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet56', 
                       choices=['resnet56', 'resnet110'],
                       help='模型架构')
    
    # 训练参数
    parser.add_argument('--client_number', type=int, default=10, help='客户端数量')
    parser.add_argument('--clients_per_round', type=int, default=0, help='每轮随机选择参与训练的客户端数量(0表示全量)')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--rounds', type=int, default=100, help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--lr_factor', type=float, default=0.9, help='学习率衰减因子')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 预训练参数
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--pretrain_lr', type=float, default=0.01, help='预训练学习率')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    return args


class AutoFlushFileHandler(logging.FileHandler):
    """自动刷新的文件处理器"""
    def emit(self, record):
        super().emit(record)
        self.flush()  # 每次写入后立即刷新


class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
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
    """SplitFed客户端模型"""
    def __init__(self, input_channels=3, model_type='resnet56'):
        super(SplitFedClientModel, self).__init__()
        
        # ResNet56: 每层9个块
        if model_type == 'resnet56':
            num_blocks = [9, 9, 9]
        elif model_type == 'resnet110':
            num_blocks = [18, 18, 18]
        else:
            num_blocks = [9, 9, 9]
        
        # 客户端包含前两层
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer 1
        self.layer1 = self._make_layer(BasicBlock, 16, 16, num_blocks[0])
        
        # Layer 2的前半部分
        client_layer2_blocks = num_blocks[1] // 2
        self.layer2_client = self._make_layer(BasicBlock, 16, 32, client_layer2_blocks, stride=2)
        
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
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
    """SplitFed服务器模型"""
    def __init__(self, num_classes=10, model_type='resnet56'):
        super(SplitFedServerModel, self).__init__()
        
        if model_type == 'resnet56':
            num_blocks = [9, 9, 9]
        elif model_type == 'resnet110':
            num_blocks = [18, 18, 18]
        else:
            num_blocks = [9, 9, 9]
        
        # Layer 2的后半部分
        server_layer2_blocks = num_blocks[1] - num_blocks[1] // 2
        self.layer2_server = self._make_layer(BasicBlock, 32, 32, server_layer2_blocks)
        
        # Layer 3
        self.layer3 = self._make_layer(BasicBlock, 32, 64, num_blocks[2], stride=2)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
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


def _simulate_resource_delay(client, measured_train_time: float, model_type: str, tier: int = 6) -> float:
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

    tier = int(tier)
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
    """SplitFed客户端"""
    def __init__(
        self,
        client_id,
        train_data,
        test_data,
        device='cuda',
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
        """设置客户端模型"""
        self.model = model.to(self.device)

    def update_resources(self, device_tier=None, compute_power=None, network_speed=None, storage_capacity=None):
        if device_tier is not None:
            self.device_tier = int(device_tier)
        if compute_power is not None:
            self.compute_power = float(compute_power)
        if network_speed is not None:
            self.network_speed = float(network_speed)
        if storage_capacity is not None:
            self.storage_capacity = int(storage_capacity)
        
    def pretrain(self, epochs, lr):
        """预训练阶段 - 本地训练，不与服务器通信"""
        if self.model is None:
            raise ValueError("Client model not set")
        
        logging.info(f"Client {self.client_id} starting pre-training...")
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播（仅客户端模型）
                features = self.model(data)
                
                # 使用简单的重构损失（鼓励学习有用特征）
                loss = torch.mean(torch.abs(features))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if (epoch + 1) % 5 == 0:
                logging.info(f"  Client {self.client_id} Pre-train Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logging.info(f"Client {self.client_id} pre-training completed, Best Loss: {best_loss:.4f}")
        
    def train_step(self, server_model, optimizer, criterion, model_type: str):
        """训练一步"""
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
            
            # 前向传播：客户端 -> 服务器
            client_features = self.model(data)
            outputs = server_model(client_features)
            
            # 计算损失
            loss = criterion(outputs, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            batch_count += 1
        
        avg_loss = total_loss / max(1, batch_count)
        accuracy = 100.0 * correct / max(1, total)

        measured_time = float(time.time() - start_time)
        simulated_delay = _simulate_resource_delay(self, measured_time, model_type=model_type, tier=6)

        return avg_loss, accuracy, measured_time, simulated_delay
    
    def evaluate(self, server_model):
        """在本地测试集上评估"""
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
                
                # 前向传播
                client_features = self.model(data)
                outputs = server_model(client_features)
                
                # 计算损失和准确率
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
    """加载数据集"""
    logging.info(f"Loading dataset: {args.dataset}")
    
    if args.dataset == "cifar10":
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    elif args.dataset == "fashion_mnist":
        data_loader = load_partition_data_fashion_mnist
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    else:
        data_loader = load_partition_data_cifar10
    
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(
        args.dataset, args.data_dir, args.partition_method,
        args.partition_alpha, args.client_number, args.batch_size
    )
    
    logging.info(f"Total training samples: {train_data_num}")
    logging.info(f"Total testing samples: {test_data_num}")
    logging.info(f"Number of classes: {class_num}")
    
    return (train_data_local_num_dict, train_data_local_dict, 
            test_data_local_dict, class_num)


def aggregate_models(client_models, client_data_sizes):
    """聚合客户端模型"""
    total_data = sum(client_data_sizes.values())
    
    # 初始化聚合模型
    aggregated_state = {}
    
    # 获取第一个客户端模型的键
    first_client_id = list(client_models.keys())[0]
    param_keys = client_models[first_client_id].state_dict().keys()
    
    # 对每个参数进行加权平均
    for key in param_keys:
        weighted_sum = None
        
        for client_id, model in client_models.items():
            weight = client_data_sizes[client_id] / total_data
            param = model.state_dict()[key]
            
            if 'num_batches_tracked' in key:
                # BatchNorm的追踪参数使用最大值
                if weighted_sum is None:
                    weighted_sum = param
                else:
                    weighted_sum = torch.max(weighted_sum, param)
            else:
                # 其他参数使用加权平均
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
        
        aggregated_state[key] = weighted_sum
    
    return aggregated_state


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建日志文件
    log_filename = f"{args.dataset}_{args.model}_splitfed_alpha{args.partition_alpha}.txt"
    
    # 配置日志 - 使用自动刷新的文件处理器
    file_handler = AutoFlushFileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 获取root logger并配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除已有的handlers
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 创建一个函数来强制刷新日志
    def flush_logs():
        for handler in logger.handlers:
            handler.flush()
    
    logging.info("="*80)
    logging.info("SplitFed: Split Federated Learning")
    logging.info("="*80)
    logging.info(f"Configuration:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    logging.info("="*80)
    flush_logs()  # 刷新日志确保配置信息写入
    
    # 设置随机种子
    set_seed(args.seed)
    logging.info(f"Random seed set to: {args.seed}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = load_dataset(args)
    
    # 确定输入通道数
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    logging.info(f"为 {args.client_number} 个客户端分配设备资源(5级)...")
    client_profiles = build_initial_client_profiles(
        client_number=args.client_number,
        train_data_local_dict=train_data_local_dict,
        n_classes=class_num,
        seed=int(args.seed),
    )
    client_resources = {
        cid: {
            "device_tier": profile.device_tier,
            "heterogeneity_score": profile.heterogeneity_score,
            "compute_power": profile.compute_power,
            "network_speed": profile.network_speed,
            "storage_capacity": profile.storage_capacity,
            "model_tier": profile.model_tier,
        }
        for cid, profile in client_profiles.items()
    }
    
    # 创建客户端
    clients = {}
    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        client = SplitFedClient(
            client_id=client_id,
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            device=device,
            device_tier=resource.get("device_tier"),
            compute_power=resource.get("compute_power"),
            network_speed=resource.get("network_speed"),
            storage_capacity=resource.get("storage_capacity"),
            heterogeneity_score=resource.get("heterogeneity_score"),
            model_tier=resource.get("model_tier"),
        )
        clients[client_id] = client
        logging.info(
            "Created Client %s, Training samples: %s, device_tier=%s(%s), compute_power=%.3f, network_speed=%s, storage_capacity=%s",
            client_id,
            train_data_local_num_dict[client_id],
            resource.get("device_tier"),
            device_tier_label(resource.get("device_tier")),
            float(resource.get("compute_power") or 0.0),
            resource.get("network_speed"),
            resource.get("storage_capacity"),
        )
    
    # 创建客户端模型
    client_models = {}
    for client_id in range(args.client_number):
        model = SplitFedClientModel(input_channels=input_channels, model_type=args.model)
        clients[client_id].set_model(model)
        client_models[client_id] = model
    
    # 创建服务器模型
    server_model = SplitFedServerModel(num_classes=class_num, model_type=args.model).to(device)
    flush_logs()  # 刷新客户端创建信息
    
    # ========== 预训练阶段 ==========
    logging.info("\n" + "="*80)
    logging.info("Starting Pre-training Phase")
    logging.info("="*80)
    
    for client_id, client in clients.items():
        client.pretrain(epochs=args.pretrain_epochs, lr=args.pretrain_lr)
        flush_logs()  # 每个客户端预训练后刷新
    
    logging.info("Pre-training Phase Completed\n")
    flush_logs()  # 预训练阶段完成后刷新
    
    # ========== 正式训练阶段 ==========
    logging.info("="*80)
    logging.info("Starting Federated Learning Training")
    logging.info("="*80)
    
    best_accuracy = 0.0
    best_round = 0
    
    for round_idx in range(args.rounds):
        round_start = time.time()
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Round {round_idx + 1}/{args.rounds}")
        logging.info(f"{'='*80}")

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
                cid
                for cid in range(args.client_number)
                if int(new_device_tiers[cid]) != int(old_device_tiers[cid])
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
        
        # 本轮的学习率
        current_lr = args.lr * (args.lr_factor ** (round_idx // 10))
        
        # 创建优化器（客户端和服务器模型一起优化）
        all_params = []
        for cid in selected_client_ids:
            all_params.extend(list(client_models[cid].parameters()))
        all_params.extend(list(server_model.parameters()))
        
        optimizer = torch.optim.Adam(all_params, lr=current_lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # 训练每个客户端
        train_losses = []
        train_accs = []
        time_costs = []
        time_costs_real = []
        training_wallclock_start = time.time()
        
        for client_id in selected_client_ids:
            client = clients[client_id]
            loss, acc, measured_time, simulated_delay = client.train_step(server_model, optimizer, criterion, model_type=args.model)
            train_losses.append(loss)
            train_accs.append(acc)
            time_costs.append(float(measured_time + simulated_delay))
            time_costs_real.append(float(measured_time))
            
            logging.info(
                "  Client %s - Train Loss: %.4f, Accuracy: %.2f%%, device_tier=%s, TrainTimeRaw: %.3fs, SimDelay: %.3fs, TrainTime: %.3fs",
                client_id,
                float(loss),
                float(acc),
                getattr(client, "device_tier", None),
                float(measured_time),
                float(simulated_delay),
                float(measured_time + simulated_delay),
            )
        training_wallclock = float(time.time() - training_wallclock_start)
        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        
        # 聚合客户端模型
        selected_models = {cid: client_models[cid] for cid in selected_client_ids}
        selected_sizes = {cid: train_data_local_num_dict[cid] for cid in selected_client_ids}
        aggregation_start = time.time()
        aggregated_state = aggregate_models(selected_models, selected_sizes)
        aggregation_time = float(time.time() - aggregation_start)
        
        # 更新所有客户端模型
        for client_id, model in client_models.items():
            model.load_state_dict(aggregated_state)
        
        # 评估
        test_losses = []
        test_accs = []
        test_samples = []
        
        for client_id, client in clients.items():
            loss, acc, samples = client.evaluate(server_model)
            test_losses.append(loss)
            test_accs.append(acc)
            test_samples.append(samples)
        
        # 计算加权平均准确率
        total_samples = sum(test_samples)
        weighted_avg_acc = sum(acc * samples for acc, samples in zip(test_accs, test_samples)) / total_samples
        avg_test_loss = np.mean(test_losses)
        
        round_time = time.time() - round_start
        training_time_real = float(sum(time_costs_real)) if time_costs_real else 0.0
        training_time_simulated = float(sum(time_costs)) if time_costs else 0.0
        
        # 记录统计信息
        logging.info(f"\nRound {round_idx + 1} Statistics:")
        logging.info(f"  Average Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  Average Train Accuracy: {avg_train_acc:.2f}%")
        logging.info(f"  Average Test Loss: {avg_test_loss:.4f}")
        logging.info(f"  Averaged Test Accuracy: {weighted_avg_acc:.2f}%")
        logging.info(f"  Learning Rate: {current_lr:.6f}")
        logging.info(
            "  轮次总时间: %.2f秒, 训练(真实墙钟): %.2f秒, 训练(真实累计): %.2f秒, 训练(模拟累计): %.2f秒, 聚合: %.2f秒",
            float(round_time),
            float(training_wallclock),
            float(training_time_real),
            float(training_time_simulated),
            float(aggregation_time),
        )
        
        # 更新最佳准确率
        if weighted_avg_acc > best_accuracy:
            best_accuracy = weighted_avg_acc
            best_round = round_idx + 1
            
            # 保存最佳模型
            torch.save({
                'round': best_round,
                'client_models': {cid: model.state_dict() for cid, model in client_models.items()},
                'server_model': server_model.state_dict(),
                'accuracy': best_accuracy,
            }, f"{args.dataset}_{args.model}_splitfed_alpha{args.partition_alpha}_best.pth")
            
            logging.info(f"  *** New Best Accuracy: {best_accuracy:.2f}% (Round {best_round}) ***")
        
        # 每轮结束后刷新日志
        flush_logs()
    
    # 训练完成
    logging.info("\n" + "="*80)
    logging.info("Training Completed!")
    logging.info(f"Best Test Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    logging.info("="*80)
    
    # 最终刷新并关闭handlers
    flush_logs()
    for handler in logger.handlers:
        handler.close()
    
    logging.info(f"\nLog file saved to: {log_filename}")


if __name__ == "__main__":
    main()
