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
import atexit
import copy
import warnings
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import math

# 设置环境变量解决MKL兼容性问题
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ.setdefault('OMP_NUM_THREADS', '1')

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入自定义模块
from model.resnet import (
    EnhancedServerModel,
    TierAwareClientModel,
    ImprovedGlobalClassifier,
    get_resnet_num_blocks,
    get_tier_shared_block_counts,
)
# 修改聚合器导入
# from utils.simplified_aggregator import SimplifiedGlobalAggregator, HierarchicalAggregator, KnowledgeDistillationAggregatorWrapper
from utils.simplified_aggregator import KnowledgeDistillationAggregatorWrapper
from utils.simplified_trainer import SimplifiedSerialTrainer
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_loss import EnhancedStagedLoss
# 添加聚类分组导入
from utils.softlabel_clustering import SoftLabelClusterManager

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist

# 添加聚类管理器导入
from utils.client_clustering import ClientClusterManager
from client_resource_allocation import (
    candidate_model_tiers_for_device_tier,
    build_initial_client_profiles,
    device_tier_label,
    filter_candidates_by_heterogeneity,
    HETEROGENEITY_THRESHOLD_A,
    HETEROGENEITY_THRESHOLD_B,
    mutate_device_tiers,
    recompute_model_tiers_from_device_and_scores,
    sample_resources_for_device_tier,
)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='简化版PSFL: 个性化拆分联邦学习框架')
    
    # 实验标识
    parser.add_argument('--running_name', default="PSFL_KnowledgeDistillation", type=str, help='实验名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.005, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.9, type=float, help='学习率衰减因子')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', help='使用的神经网络 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='fashion_mnist', 
                       help='训练数据集 (cifar10, cifar100, svhn, fashion_mnist, cinic10)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha（固定为0.5，50%异质性）')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=1, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=10, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=256, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')

    parser.add_argument('--clients_per_round', type=int, default=0, help='每轮随机选择参与训练的客户端数量(0表示全量)')
    parser.add_argument('--split_strategy', type=str, default='budgeted', choices=['heuristic', 'budgeted'], help='动态拆分点策略')
    parser.add_argument('--split_time_budget', type=float, default=0.0, help='每轮训练时间预算(秒, 0表示自动预算)')
    parser.add_argument('--split_budget_scale', type=float, default=1.0, help='自动预算倍率(基于最快方案估计)')
    parser.add_argument('--split_acc_gain_per_tier', type=float, default=0.2, help='tier每降低1的预期精度增益(用于估计)')
    parser.add_argument('--split_ema_decay', type=float, default=0.7, help='时间/精度历史EMA衰减')
    
    # PSFL特有参数
    parser.add_argument('--init_alpha', default=0.6, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.15, type=float, help='初始特征对齐损失权重')
    
    # 知识蒸馏相关参数
    parser.add_argument('--distillation_temperature', type=float, default=4.0,
                       help='知识蒸馏温度参数')
    parser.add_argument('--distillation_alpha', type=float, default=0.3,
                       help='真实标签损失权重')
    parser.add_argument('--distillation_beta', type=float, default=0.7,
                       help='蒸馏损失权重')
    parser.add_argument('--distillation_epochs', type=int, default=3,
                       help='蒸馏训练轮数')
    
    # 聚类相关参数
    parser.add_argument('--num_clusters', type=int, default=3, 
                       help='客户端聚类数量')
    parser.add_argument('--clustering_method', type=str, default='cosine_similarity',
                       choices=['cosine_similarity', 'euclidean'],
                       help='聚类相似度度量方法')
    
    # 新增：预热训练参数
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='客户端预热训练轮数')
    parser.add_argument('--warmup_lr', type=float, default=0.01,
                       help='预热训练学习率')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='预热训练早停耐心值')
    
    args = parser.parse_args()
    return args


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self._streams:
            fn = getattr(s, "isatty", None)
            if fn and fn():
                return True
        return False


def _build_console_log_filename(args) -> str:
    alpha_str = str(args.partition_alpha).replace(".", "p")
    return f"{args.dataset}_clients{args.client_number}_{args.model}_alpha{alpha_str}.txt"


def setup_logging(args):
    log_file = _build_console_log_filename(args)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fh = open(log_file, "a", encoding="utf-8", buffering=1)
    def _restore_and_close():
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
    atexit.register(_restore_and_close)
    sys.stdout = _TeeStream(original_stdout, log_fh)
    sys.stderr = _TeeStream(original_stderr, log_fh)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("PSFL")
    logger.info(f"控制台输出将同时写入: {log_file}")
    return logger

# 替换原有的load_dataset函数：

def load_dataset(args):
    
    # 根据数据集类型选择相应的数据加载器
    if args.dataset == "cifar10":
        from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
        data_loader = load_partition_data_cifar100
    elif args.dataset == "fashion_mnist":
        from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist
        data_loader = load_partition_data_fashion_mnist
    elif args.dataset == "cinic10":
        from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    elif args.dataset == "svhn":
        from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
        data_loader = load_partition_data_svhn
    else:
        # 默认使用CIFAR-10
        from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
        data_loader = load_partition_data_cifar10
    
    # 加载数据（不带聚类）
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(
        args.dataset, args.data_dir, args.partition_method,
        args.partition_alpha, args.client_number, args.batch_size
    )
    
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset

def allocate_device_resources(client_number):
    resources = {}
    
    # 随机分配tier (1-7)，1为高性能，7为低性能
    tier_weights = [0.12, 0.14, 0.16, 0.16, 0.16, 0.14, 0.12]
    tiers = random.choices(range(1, 8), weights=tier_weights, k=client_number)
    
    # 为每个客户端分配资源
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier == 1:  # 高性能设备
            compute_power = random.uniform(0.8, 1.0)
            network_speed = random.choice([50, 100, 200])
            storage_capacity = random.choice([256, 512, 1024])
        elif tier == 2:  # 中高性能设备
            compute_power = random.uniform(0.6, 0.8)
            network_speed = random.choice([30, 50, 100])
            storage_capacity = random.choice([128, 256, 512])
        elif tier == 3:  # 中等设备
            compute_power = random.uniform(0.45, 0.65)
            network_speed = random.choice([20, 30, 50])
            storage_capacity = random.choice([96, 128, 256])
        elif tier == 4:  # 中低性能设备
            compute_power = random.uniform(0.3, 0.5)
            network_speed = random.choice([20, 30, 50])
            storage_capacity = random.choice([64, 128, 256])
        elif tier == 5:  # 低性能设备
            compute_power = random.uniform(0.2, 0.35)
            network_speed = random.choice([10, 20, 30])
            storage_capacity = random.choice([32, 64, 128])
        elif tier == 6:  # 更低性能设备
            compute_power = random.uniform(0.12, 0.25)
            network_speed = random.choice([5, 10, 20])
            storage_capacity = random.choice([16, 32, 64])
        else:  # tier 7, 极低性能设备
            compute_power = random.uniform(0.05, 0.15)
            network_speed = random.choice([2, 5, 10])
            storage_capacity = random.choice([8, 16, 32])
        
        # 存储资源信息
        resources[client_id] = {
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources


def _tier_label(tier: int) -> str:
    tier = int(tier)
    labels = {
        1: "高性能设备",
        2: "中高性能设备",
        3: "中等设备",
        4: "中低性能设备",
        5: "低性能设备",
        6: "更低性能设备",
        7: "极低性能设备",
    }
    return labels.get(tier, "未知等级")


def _format_split_assignment(model_type: str, tier: int) -> str:
    num_blocks = get_resnet_num_blocks(model_type)
    shared = get_tier_shared_block_counts(model_type=model_type, tier=tier)
    l1, l2, l3 = num_blocks
    s1, s2, s3 = shared["layer1"], shared["layer2"], shared["layer3"]

    def _fmt_range(start: int, end: int) -> str:
        if start > end:
            return "none"
        return f"{start}..{end}"

    if s1 < l1:
        boundary = (
            f"layer1: client {_fmt_range(0, s1-1)}, "
            f"server {_fmt_range(s1, l1-1)}"
        )
    elif s2 < l2:
        boundary = (
            f"layer2: client {_fmt_range(0, s2-1)}, "
            f"server {_fmt_range(s2, l2-1)}"
        )
    elif s3 < l3:
        boundary = (
            f"layer3: client {_fmt_range(0, s3-1)}, "
            f"server {_fmt_range(s3, l3-1)}"
        )
    else:
        boundary = "backbone: client full, server classifier-only"

    return (
        f"shared(layer1={s1}/{l1}, layer2={s2}/{l2}, layer3={s3}/{l3}); "
        f"split={boundary}"
    )


class BudgetedSplitPointAllocator:
    def __init__(
        self,
        *,
        client_data_sizes,
        model_type: str,
        ema_decay: float,
        acc_gain_per_tier: float,
        logger,
    ):
        self.client_data_sizes = dict(client_data_sizes)
        self.model_type = str(model_type)
        self.ema_decay = float(ema_decay)
        self.acc_gain_per_tier = float(acc_gain_per_tier)
        self.logger = logger
        self.client_stats = {}
        self.global_measured_time_per_sample = None

    def update_client_observation(self, *, client_id: int, tier: int, raw_time_cost: float, train_accuracy=None):
        client_id = int(client_id)
        tier = int(tier)
        raw_time_cost = float(raw_time_cost or 0.0)
        key = client_id
        prev = self.client_stats.get(key, {})

        data_size = float(self.client_data_sizes.get(client_id, 1))
        measured_per_sample = raw_time_cost / max(1.0, data_size)

        if prev.get("measured_time_per_sample") is None:
            prev["measured_time_per_sample"] = measured_per_sample
        else:
            prev["measured_time_per_sample"] = (
                self.ema_decay * float(prev["measured_time_per_sample"])
                + (1.0 - self.ema_decay) * measured_per_sample
            )

        if train_accuracy is not None:
            train_accuracy = float(train_accuracy)
            if prev.get("train_accuracy") is None:
                prev["train_accuracy"] = train_accuracy
            else:
                prev["train_accuracy"] = (
                    self.ema_decay * float(prev["train_accuracy"])
                    + (1.0 - self.ema_decay) * train_accuracy
                )

        prev["last_tier"] = tier
        self.client_stats[key] = prev

        if self.global_measured_time_per_sample is None:
            self.global_measured_time_per_sample = measured_per_sample
        else:
            self.global_measured_time_per_sample = (
                0.9 * float(self.global_measured_time_per_sample) + 0.1 * measured_per_sample
            )

    def allocate(self, *, selected_client_ids, client_resources, budget_seconds: float, candidate_tiers_by_client):
        selected_client_ids = [int(x) for x in selected_client_ids]
        budget_seconds = float(budget_seconds)

        baseline_choice = {}
        baseline_time = {}
        baseline_value = {}

        for client_id in selected_client_ids:
            candidates = list(candidate_tiers_by_client[client_id])
            base_tier = max(int(t) for t in candidates)
            baseline_choice[client_id] = base_tier

            t_est, v_est = self._estimate_time_and_value(
                client_id=client_id,
                tier=base_tier,
                client_resources=client_resources,
            )
            baseline_time[client_id] = t_est
            baseline_value[client_id] = v_est

        fastest_sum_time = float(sum(baseline_time.values()))
        effective_budget = float(max(fastest_sum_time, budget_seconds))

        chosen = dict(baseline_choice)
        used_time = float(fastest_sum_time)

        if used_time >= effective_budget:
            self._log_allocation(selected_client_ids, chosen, baseline_time, effective_budget, used_time)
            return chosen

        upgrade_items = []
        for client_id in selected_client_ids:
            base_tier = baseline_choice[client_id]
            candidates = sorted({int(t) for t in candidate_tiers_by_client[client_id]})
            for tier in candidates:
                if tier >= base_tier:
                    continue
                alt_time, alt_value = self._estimate_time_and_value(
                    client_id=client_id,
                    tier=tier,
                    client_resources=client_resources,
                )
                dt = float(alt_time - baseline_time[client_id])
                dv = float(alt_value - baseline_value[client_id])
                if dt <= 0.0 or dv <= 0.0:
                    continue
                upgrade_items.append((dv / dt, dv, dt, client_id, tier, alt_time))

        upgrade_items.sort(reverse=True, key=lambda x: float(x[0]))
        upgraded = set()
        for _, _, dt, client_id, tier, alt_time in upgrade_items:
            if client_id in upgraded:
                continue
            if used_time + dt > effective_budget:
                continue
            chosen[client_id] = int(tier)
            baseline_time[client_id] = float(alt_time)
            used_time += float(dt)
            upgraded.add(client_id)

        self._log_allocation(selected_client_ids, chosen, baseline_time, effective_budget, used_time)
        return chosen

    def _log_allocation(self, selected_client_ids, chosen, chosen_time_est, budget_seconds, used_seconds):
        if not self.logger:
            return
        tiers = {int(cid): int(chosen[cid]) for cid in selected_client_ids}
        self.logger.info(
            "拆分点预算策略: 参与=%s, 预算=%.3fs, 预计=%.3fs, 分配=%s",
            len(selected_client_ids),
            float(budget_seconds),
            float(used_seconds),
            tiers,
        )

    def _estimate_time_and_value(self, *, client_id: int, tier: int, client_resources):
        client_id = int(client_id)
        tier = int(tier)
        data_size = float(self.client_data_sizes.get(client_id, 1))
        base_per_sample = self.client_stats.get(client_id, {}).get("measured_time_per_sample")
        if base_per_sample is None:
            base_per_sample = self.global_measured_time_per_sample
        if base_per_sample is None:
            base_per_sample = 1.0e-4

        measured_time = float(base_per_sample) * max(1.0, data_size)
        simulated_delay = self._simulate_delay_est(client_resources=client_resources[client_id], measured_train_time=measured_time, tier=tier)
        total_time = float(measured_time + simulated_delay)

        prev = self.client_stats.get(client_id, {})
        last_acc = prev.get("train_accuracy")
        last_tier = prev.get("last_tier")
        if last_acc is None or last_tier is None:
            acc_est = 50.0 - float(self.acc_gain_per_tier) * float(tier - 4)
        else:
            acc_est = float(last_acc) + float(self.acc_gain_per_tier) * float(int(last_tier) - int(tier))

        heterogeneity = float(client_resources[client_id].get("heterogeneity_score", 0.0) or 0.0)
        value = float(acc_est) * max(1.0, data_size) * (1.0 + 0.5 * heterogeneity)
        return total_time, value

    def _simulate_delay_est(self, *, client_resources, measured_train_time: float, tier: int) -> float:
        try:
            compute_power = float(client_resources.get("compute_power", 1.0) or 1.0)
        except Exception:
            compute_power = 1.0
        try:
            network_speed = float(client_resources.get("network_speed", 50.0) or 50.0)
        except Exception:
            network_speed = 50.0

        compute_power = max(0.05, min(1.0, compute_power))
        network_speed = max(1.0, network_speed)
        measured_train_time = float(max(0.0, measured_train_time))

        num_blocks = get_resnet_num_blocks(self.model_type)
        total_blocks = float(max(1, sum(int(x) for x in num_blocks)))
        shared = get_tier_shared_block_counts(model_type=self.model_type, tier=int(tier))
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


def decide_split_points_and_rebuild_models(
    *,
    client_ids,
    args,
    client_manager,
    client_resources,
    client_models,
    server_models,
    global_shared_layers,
    global_server_model,
    num_classes: int,
    input_channels: int,
    device,
    logger,
    tier_overrides=None,
):
    for client_id in client_ids:
        info = client_resources[client_id]
        device_tier = int(info["device_tier"])
        heterogeneity_score = float(info["heterogeneity_score"])

        candidates = candidate_model_tiers_for_device_tier(device_tier)
        filtered = filter_candidates_by_heterogeneity(candidates, heterogeneity_score)
        if tier_overrides is not None and client_id in tier_overrides:
            chosen_tier = int(tier_overrides[client_id])
        else:
            chosen_tier = int(max(filtered)) if filtered else int(max(candidates)) if candidates else 4

        info["tier"] = chosen_tier
        client_manager.update_client_tier(client_id, chosen_tier)

        old_client_state = None
        if client_id in client_models and client_models[client_id] is not None:
            try:
                old_client_state = client_models[client_id].state_dict()
            except Exception:
                old_client_state = None

        client_models[client_id] = TierAwareClientModel(
            num_classes=num_classes,
            tier=chosen_tier,
            model_type=args.model,
            input_channels=input_channels,
        )
        if old_client_state is not None:
            client_models[client_id].load_state_dict(old_client_state, strict=False)
        client_models[client_id].load_state_dict(global_shared_layers, strict=False)

        server_models[client_id] = EnhancedServerModel(
            num_classes=num_classes,
            tier=chosen_tier,
            model_type=args.model,
            input_channels=input_channels,
        )
        server_models[client_id].load_state_dict(global_server_model, strict=False)

        split_info = _format_split_assignment(args.model, chosen_tier)
        logger.info(
            "重建客户端模型: client=%s, device_tier=%s(%s), heterogeneity=%.4f, candidates=%s, filtered=%s, split_tier=%s, %s",
            client_id,
            device_tier,
            device_tier_label(device_tier),
            heterogeneity_score,
            candidates,
            filtered,
            chosen_tier,
            split_info,
        )

def load_global_test_set(args):
    """创建完整的全局测试集用于评估泛化性能"""
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                [0.2675, 0.2565, 0.2761])
        ])
        
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "svhn":
        from api.data_preprocessing.svhn.datasets import SVHN_truncated

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
            ]
        )

        testset = SVHN_truncated(root=args.data_dir, train=False, download=True, transform=transform_test)
        
    elif args.dataset == "fashion_mnist":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530])
        ])
        
        testset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform_test)
            
    elif args.dataset == "cinic10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform_test)
    else:
        # 默认返回CIFAR10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    
    # 创建数据加载器 - 使用完整的测试集
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return test_loader

def evaluate_global_model(client_models, server_models, global_test_loader, device, client_number):
    """
    评估全局模型在全局测试集上的性能
    测试所有客户端模型并取平均值
    """
    all_accuracies = []
    
    for client_id in range(client_number):
        # 获取客户端模型
        client_model = client_models[client_id].to(device)
        server_model = server_models[client_id].to(device)
        
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in global_test_loader:
                # 移到设备
                data, target = data.to(device), target.to(device)
                
                try:
                    # 完整的前向传播：客户端共享层 -> 服务器模型 -> 分类结果
                    _, shared_features, _ = client_model(data)
                    logits = server_model(shared_features)
                    
                    _, predicted = logits.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                except Exception as e:
                    logging.error(f"评估客户端 {client_id} 时出现错误: {str(e)}")
                    continue
        
        if total > 0:
            accuracy = 100.0 * correct / total
            all_accuracies.append(accuracy)
            logging.info(f"客户端 {client_id} 全局测试集准确率: {accuracy:.2f}%")
    
    # 计算平均准确率
    if all_accuracies:
        avg_accuracy = sum(all_accuracies) / len(all_accuracies)
        logging.info(f"所有客户端平均全局测试集准确率: {avg_accuracy:.2f}%")
        return avg_accuracy
    else:
        logging.warning("没有成功评估任何客户端")
        return 0.0


def evaluate_split_model_local_weighted(client_manager, client_models, server_models, device, client_number):
    total_correct = 0
    total_samples = 0
    per_client = []

    for client_id in range(int(client_number)):
        client = client_manager.get_client(client_id)
        if not client:
            continue

        client_model = client_models[client_id].to(device)
        server_model = server_models[client_id].to(device)

        client_model.eval()
        server_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in client.test_data:
                data, target = data.to(device), target.to(device)
                try:
                    shared_features = client_model.shared_base(data)
                except Exception:
                    _, shared_features, _ = client_model(data)
                logits = server_model(shared_features)
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        if total > 0:
            acc = 100.0 * correct / total
            per_client.append((client_id, acc, total))
            total_correct += correct
            total_samples += total

    for client_id, acc, total in per_client:
        logging.info(f"客户端 {client_id} 拆分模型本地测试集准确率: {acc:.2f}% (n={total})")

    if total_samples > 0:
        avg_accuracy = 100.0 * total_correct / total_samples
        logging.info(f"所有客户端加权平均拆分模型本地测试集准确率: {avg_accuracy:.2f}% (n={total_samples})")
        return avg_accuracy

    logging.warning("没有成功评估任何客户端的拆分模型(本地测试集)")
    return 0.0

def create_aggregator(args, client_data_sizes, cluster_manager, device):
    """创建知识蒸馏聚合器"""
    
    if cluster_manager is None:
        logging.error("知识蒸馏聚合需要聚类管理器")
        raise ValueError("知识蒸馏聚合需要聚类管理器")
        
    logging.info("创建知识蒸馏聚合器...")
    kd_aggregator = KnowledgeDistillationAggregatorWrapper(
        client_data_sizes=client_data_sizes,
        cluster_manager=cluster_manager,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        device=device
    )
    
    # 配置蒸馏参数
    kd_aggregator.update_distillation_config(
        temperature=args.distillation_temperature,
        alpha=args.distillation_alpha,
        beta=args.distillation_beta
    )
    
    return kd_aggregator

# 独立的客户端聚类流程
def perform_client_clustering(args, client_models, train_data_local_dict, 
                              test_data_local_dict, device, logger):
    """
    执行客户端聚类（最佳方案：本地测试集 + 软标签 + 分类器权重）
    
    Args:
        args: 命令行参数
        client_models: 客户端模型字典
        train_data_local_dict: 训练数据字典
        test_data_local_dict: 测试数据字典
        device: 设备
        logger: 日志记录器
        
    Returns:
        cluster_manager: 聚类管理器（用于创建聚合器）
    """
    logger.info("\n" + "="*80)
    logger.info("开始客户端聚类流程（最佳方案）")
    logger.info("="*80)
    
    # 确定类别数和输入通道数
    if args.dataset == "cifar10":
        num_classes = 10
        input_channels = 3
    elif args.dataset == "cifar100":
        num_classes = 100
        input_channels = 3
    elif args.dataset == "fashion_mnist":
        num_classes = 10
        input_channels = 1
    elif args.dataset == "cinic10":
        num_classes = 10
        input_channels = 3
    else:
        num_classes = 10
        input_channels = 3
    
    # 步骤1: 创建软标签聚类管理器
    logger.info("\n步骤1: 创建软标签聚类管理器")
    softlabel_manager = SoftLabelClusterManager(
        num_classes=num_classes,
        device=device
    )
    
    # 步骤2: 预热训练客户端模型
    logger.info("\n步骤2: 预热训练客户端模型")
    logger.info(f"预热轮数: {args.warmup_epochs}")
    
    trained_models = softlabel_manager.warmup_train_clients(
        client_models=client_models,
        train_data_local_dict=train_data_local_dict,
        test_data_local_dict=test_data_local_dict,
        warmup_epochs=args.warmup_epochs,
        lr=args.warmup_lr,
        early_stop_patience=args.early_stop_patience
    )
    
    # 步骤3: 在本地测试集上收集软标签（最佳方案）
    logger.info("\n步骤3: 在本地测试集上收集软标签")
    softlabel_manager.collect_soft_labels_from_local_testset(
        client_models=trained_models,
        test_data_local_dict=test_data_local_dict
    )
    
    # 步骤4: 提取分类器为中心的特征（最佳方案）
    logger.info("\n步骤4: 提取分类器为中心的特征")
    softlabel_manager.extract_classifier_focused_features(trained_models)
    
    # 步骤5: 执行聚类
    logger.info("\n步骤5: 执行客户端聚类")
    cluster_assignments = softlabel_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method=args.clustering_method
    )
    
    logger.info(f"\n聚类结果: {cluster_assignments}")
    
    # 步骤6: 创建兼容的聚类管理器（用于知识蒸馏聚合）
    logger.info("\n步骤6: 创建聚类管理器")
    
    cluster_manager = ClientClusterManager(num_classes=num_classes)
    
    # 手动设置聚类结果
    cluster_manager.cluster_assignments = cluster_assignments
    cluster_manager.cluster_info = softlabel_manager.get_cluster_info()
    
    logger.info("="*80)
    logger.info("客户端聚类完成")
    logger.info("="*80 + "\n")
    
    return cluster_manager


# 主函数
def main():
    """主函数，简化版PSFL实现"""
    # 解析命令行参数
    args = parse_arguments()

    args.initial_phase_rounds = 10
    args.alternating_phase_rounds = 70
    args.fine_tuning_phase_rounds = 20
    
    # 设置随机种子
    set_seed(int(args.seed))
    
    # 设置日志
    logger = setup_logging(args)
    # logger.info(f"初始化PSFL: 简化版本 - 聚合策略: {args.aggregation_strategy}")
    logger.info(f"数据异质性参数固定为: {args.partition_alpha} (50%异质性)")
    
    # 设置默认设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"默认设备: {device}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    
    # 获取数据集信息
    train_data_num, test_data_num, _, _, train_data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, class_num = dataset
    
    logger.info(f"数据加载完成")
    logger.info(f"训练样本总数: {train_data_num}")
    logger.info(f"测试样本总数: {test_data_num}")
    logger.info(f"类别数: {class_num}")

    # 加载全局测试集
    logger.info("加载全局IID测试集用于评估泛化性能...")
    global_test_loader = load_global_test_set(args)
    
    logger.info(f"为 {args.client_number} 个客户端分配设备资源(5级)与模型拆分tier(1-7)...")
    logger.info(
        "异质性筛选阈值: a=%.2f(低异质), b=%.2f(极端异质)",
        float(HETEROGENEITY_THRESHOLD_A),
        float(HETEROGENEITY_THRESHOLD_B),
    )
    client_profiles = build_initial_client_profiles(
        client_number=args.client_number,
        train_data_local_dict=train_data_local_dict,
        n_classes=class_num,
        seed=int(args.seed),
    )
    client_resources = {
        cid: {
            "device_tier": profile.device_tier,
            "tier": profile.model_tier,
            "heterogeneity_score": profile.heterogeneity_score,
            "entropy_norm": profile.entropy_norm,
            "gini": profile.gini,
            "compute_power": profile.compute_power,
            "network_speed": profile.network_speed,
            "storage_capacity": profile.storage_capacity,
        }
        for cid, profile in client_profiles.items()
    }
    
    # 创建客户端管理器
    logger.info("创建客户端管理器...")
    client_manager = TierHFLClientManager()
    
    # 注册客户端
    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        tier = resource["tier"]
        
        # 创建客户端
        client = client_manager.add_client(
            client_id=client_id,
            tier=tier,
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            device=device,
            lr=args.lr,
            local_epochs=args.client_epoch,
            device_tier=resource.get("device_tier"),
            compute_power=resource.get("compute_power"),
            network_speed=resource.get("network_speed"),
            storage_capacity=resource.get("storage_capacity"),
        )
        
        logger.info(f"客户端 {client_id} - Tier: {tier}, 训练样本数: {train_data_local_num_dict[client_id]}")

    # 确定输入通道数
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    # 创建客户端模型（每个客户端一个）
    logger.info(f"创建 {args.client_number} 个双路径客户端模型...")
    client_models = {}
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        split_info = _format_split_assignment(args.model, tier)
        candidates = candidate_model_tiers_for_device_tier(resource["device_tier"])
        filtered_candidates = filter_candidates_by_heterogeneity(candidates, resource["heterogeneity_score"])
        logger.info(
            "客户端 %s 分配模型: device_tier=%s(%s), heterogeneity=%.4f, candidates=%s, filtered=%s, tier=%s(%s), compute_power=%.3f, network_speed=%s, storage_capacity=%s, %s",
            client_id,
            resource["device_tier"],
            device_tier_label(resource["device_tier"]),
            resource["heterogeneity_score"],
            candidates,
            filtered_candidates,
            tier,
            _tier_label(tier),
            resource["compute_power"],
            resource["network_speed"],
            resource["storage_capacity"],
            split_info,
        )
        client_models[client_id] = TierAwareClientModel(
            num_classes=class_num, 
            tier=tier,
            model_type=args.model,
            input_channels=input_channels
        )

    # ========== 新增：独立的客户端聚类流程 ==========
    logger.info("\n" + "="*80)
    logger.info("执行客户端聚类")
    logger.info("="*80)
    
    cluster_manager = perform_client_clustering(
        args=args,
        client_models=client_models,
        train_data_local_dict=train_data_local_dict,
        test_data_local_dict=test_data_local_dict,
        device=device,
        logger=logger
    )
    # ========== 聚类流程结束 ==========

    # 创建服务器模型（每个客户端对应一个，包含分类器）
    logger.info(f"创建 {args.client_number} 个服务器模型...")
    server_models = {}
    for client_id in range(args.client_number):
        tier = client_resources[client_id]["tier"]
        split_info = _format_split_assignment(args.model, tier)
        logger.info(
            "客户端 %s 分配服务器端: device_tier=%s(%s), heterogeneity=%.4f, tier=%s(%s), %s",
            client_id,
            client_resources[client_id]["device_tier"],
            device_tier_label(client_resources[client_id]["device_tier"]),
            client_resources[client_id]["heterogeneity_score"],
            tier,
            _tier_label(tier),
            split_info,
        )
        server_models[client_id] = EnhancedServerModel(
            num_classes=class_num,
            tier=tier,
            model_type=args.model,
            input_channels=input_channels
        ).to(device)
    
    # 创建聚合器（根据策略选择）
    global_aggregator = create_aggregator(args, train_data_local_num_dict, cluster_manager, device)

    # 创建简化版串行训练器
    logger.info("创建简化版串行训练器...")
    trainer = SimplifiedSerialTrainer(
        client_manager=client_manager,
        client_models=client_models,
        server_models=server_models,
        device=device
    )

    split_allocator = BudgetedSplitPointAllocator(
        client_data_sizes=train_data_local_num_dict,
        model_type=args.model,
        ema_decay=float(args.split_ema_decay),
        acc_gain_per_tier=float(args.split_acc_gain_per_tier),
        logger=logger,
    )
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    last_global_shared_layers = None
    last_global_server_model = None
    
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")

        clients_per_round = int(args.clients_per_round or 0)
        if clients_per_round <= 0 or clients_per_round > int(args.client_number):
            clients_per_round = int(args.client_number)
        selection_seed_base = 424242 + int(args.seed) * 1000
        round_rng = random.Random(selection_seed_base + round_idx)
        selected_clients = round_rng.sample(range(int(args.client_number)), k=clients_per_round)
        selected_clients.sort()
        logger.info("本轮参与训练客户端: %s", selected_clients)

        if (round_idx + 1) % 20 == 0:
            old_device_tiers = {cid: int(info["device_tier"]) for cid, info in client_resources.items()}
            mutation_seed = selection_seed_base + (round_idx + 1)
            new_device_tiers = mutate_device_tiers(old_device_tiers, fraction=0.30, seed=mutation_seed)

            changed = [
                cid
                for cid in range(args.client_number)
                if int(new_device_tiers[cid]) != int(old_device_tiers[cid])
            ]
            logger.info(
                "每20轮触发动态设备变更: 选择 %s/%s 个客户端 (30%%)",
                len(changed),
                args.client_number,
            )

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
                client_manager.update_client_resources(
                    client_id,
                    device_tier=new_device,
                    compute_power=compute_power,
                    network_speed=network_speed,
                    storage_capacity=storage_capacity,
                )
                logger.info(
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

        if round_idx >= 1 and last_global_shared_layers is not None and last_global_server_model is not None:
            logger.info("本轮训练前执行拆分点决策与模型重建(使用上一轮聚合全局参数)...")
            candidate_tiers_by_client = {}
            for cid in selected_clients:
                device_tier = int(client_resources[cid]["device_tier"])
                heterogeneity_score = float(client_resources[cid]["heterogeneity_score"])
                candidates = candidate_model_tiers_for_device_tier(device_tier)
                filtered = filter_candidates_by_heterogeneity(candidates, heterogeneity_score)
                candidate_tiers_by_client[cid] = filtered or candidates or [4]

            tier_overrides = None
            if str(args.split_strategy) == "budgeted":
                budget_seconds = float(args.split_time_budget or 0.0)
                if budget_seconds <= 0.0:
                    fastest_sum = 0.0
                    for cid in selected_clients:
                        base_tier = max(int(t) for t in candidate_tiers_by_client[cid])
                        t_est, _ = split_allocator._estimate_time_and_value(
                            client_id=cid, tier=base_tier, client_resources=client_resources
                        )
                        fastest_sum += float(t_est)
                    budget_seconds = float(fastest_sum) * float(args.split_budget_scale or 1.0)

                tier_overrides = split_allocator.allocate(
                    selected_client_ids=selected_clients,
                    client_resources=client_resources,
                    budget_seconds=budget_seconds,
                    candidate_tiers_by_client=candidate_tiers_by_client,
                )

            decide_split_points_and_rebuild_models(
                client_ids=selected_clients,
                args=args,
                client_manager=client_manager,
                client_resources=client_resources,
                client_models=client_models,
                server_models=server_models,
                global_shared_layers=last_global_shared_layers,
                global_server_model=last_global_server_model,
                num_classes=class_num,
                input_channels=input_channels,
                device=device,
                logger=logger,
                tier_overrides=tier_overrides,
            )
            trainer.client_models = client_models
            trainer.server_models = server_models
        
        if round_idx < int(args.initial_phase_rounds):
            training_phase = "initial"
            logger.info("当前处于初始特征学习阶段")
        elif round_idx < int(args.initial_phase_rounds) + int(args.alternating_phase_rounds):
            training_phase = "alternating"
            logger.info("当前处于交替训练阶段")
        else:
            training_phase = "fine_tuning"
            logger.info("当前处于精细调整阶段")

        # 执行训练
        train_results, eval_results, shared_states, server_states, training_time = trainer.execute_round(
            round_idx=round_idx, 
            total_rounds=args.rounds,
            selected_client_ids=selected_clients,
            training_phase=training_phase,
        )

        simulated_training_time = float(
            sum(float(train_results[cid].get("simulated_time_cost", train_results[cid].get("time_cost", 0.0)) or 0.0) for cid in train_results)
        )

        for cid, result in train_results.items():
            client = client_manager.get_client(cid)
            tier = int(getattr(client, "tier", client_resources[cid].get("tier", 4)) or 4)
            raw_time_cost = float(result.get("raw_time_cost", result.get("time_cost", 0.0)) or 0.0)
            train_acc = result.get("local_accuracy", None)
            if train_acc is None:
                train_acc = result.get("global_accuracy", None)
            split_allocator.update_client_observation(
                client_id=int(cid),
                tier=tier,
                raw_time_cost=raw_time_cost,
                train_accuracy=(float(train_acc) if train_acc is not None else None),
            )
        
        # 全局聚合
        logger.info("执行全局聚合")
        aggregation_start_time = time.time()
        
        # 聚合共享层和服务器模型
        global_shared_layers, global_server_model = global_aggregator.aggregate(
            shared_states, server_states
        )

        # 添加调试信息
        logger.info(f"\n聚合后的全局模型统计:")
        logger.info(f"  共享层键数: {len(global_shared_layers)}")
        logger.info(f"  服务器模型键数: {len(global_server_model)}")

        # 检查是否包含 BatchNorm 的 buffers
        bn_buffers_in_shared = [k for k in global_shared_layers.keys() if 'running_mean' in k or 'running_var' in k]
        bn_buffers_in_server = [k for k in global_server_model.keys() if 'running_mean' in k or 'running_var' in k]
        logger.info(f"  共享层 BN buffers: {len(bn_buffers_in_shared)}")
        logger.info(f"  服务器模型 BN buffers: {len(bn_buffers_in_server)}")
        
        last_global_shared_layers = global_shared_layers
        last_global_server_model = global_server_model
        
        aggregation_time = time.time() - aggregation_start_time

        try:
            trainer.update_global_models(global_shared_layers, global_server_model)
        except Exception as e:
            logger.error(f"使用本轮聚合结果更新客户端共享层/服务器模型失败: {str(e)}")
        
        # 评估全局模型性能（所有客户端平均准确率）
        global_model_accuracy = evaluate_global_model(
            client_models, server_models, global_test_loader, device, args.client_number
        )

        split_model_weighted_test_accuracy = evaluate_split_model_local_weighted(
            client_manager=client_manager,
            client_models=client_models,
            server_models=server_models,
            device=device,
            client_number=args.client_number,
        )
                
        # 计算平均准确率
        avg_local_train_acc = np.mean([result.get('local_train_accuracy', 0) for result in eval_results.values()])
        avg_local_test_acc = np.mean([result.get('local_test_accuracy', 0) for result in eval_results.values()])
        avg_split_train_acc = np.mean([result.get('split_train_accuracy', 0) for result in eval_results.values()])
        avg_split_test_acc = split_model_weighted_test_accuracy
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            try:
                torch.save({
                    'client_models': {cid: model.state_dict() for cid, model in client_models.items()},
                    'server_models': {cid: model.state_dict() for cid, model in server_models.items()},
                    'global_shared_layers': global_shared_layers,
                    'global_server_model': global_server_model,
                    'round': round_idx,
                    'accuracy': best_accuracy,
                    # 'aggregation_strategy': args.aggregation_strategy
                }, f"{args.running_name}_best_model.pth")
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"本地模型 - 训练准确率: {avg_local_train_acc:.2f}%, 测试准确率: {avg_local_test_acc:.2f}%")
        logger.info(f"拆分模型 - 训练准确率: {avg_split_train_acc:.2f}%, 测试准确率: {avg_split_test_acc:.2f}%")
        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}% ")
        logger.info(f"最佳准确率: {best_accuracy:.2f}%")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练(真实墙钟): {training_time:.2f}秒, 训练(模拟累计): {simulated_training_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 动态学习率调整
        if round_idx > 0 and round_idx % 10 == 0:
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    client.lr *= args.lr_factor
                    logger.info(f"客户端 {client_id} 学习率更新为: {client.lr:.6f}")
    
    # 训练完成
    logger.info(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()
