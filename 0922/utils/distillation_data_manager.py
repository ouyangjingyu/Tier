import torch
import torch.utils.data as data
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from collections import defaultdict

class DistillationDataManager:
    """蒸馏数据管理器，从原始数据集中构建独立的蒸馏数据集"""
    
    def __init__(self, dataset_name, data_dir, distillation_ratio=0.15, batch_size=128):
        """
        Args:
            dataset_name: 数据集名称 (cifar10, fashion_mnist)
            data_dir: 数据目录
            distillation_ratio: 蒸馏数据占原始训练集的比例
            batch_size: 蒸馏数据的批次大小
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.distillation_ratio = distillation_ratio
        self.batch_size = batch_size
        
        self.distillation_loader = None
        self.validation_loader = None
        
        # 设置日志
        self.logger = logging.getLogger("DistillationDataManager")
        
    def create_distillation_dataset(self, avoid_test_indices=None):
        """
        创建蒸馏数据集，确保与测试集无重叠
        
        Args:
            avoid_test_indices: 需要避免的测试集索引，确保无数据泄漏
        """
        self.logger.info(f"为{self.dataset_name}创建蒸馏数据集...")
        
        if self.dataset_name == "cifar10":
            return self._create_cifar10_distillation_dataset(avoid_test_indices)
        elif self.dataset_name == "fashion_mnist":
            return self._create_fashion_mnist_distillation_dataset(avoid_test_indices)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
    
    def _create_cifar10_distillation_dataset(self, avoid_test_indices):
        """创建CIFAR-10蒸馏数据集"""
        from api.data_preprocessing.cifar10.datasets import CIFAR10_truncated
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                               [0.24703233, 0.24348505, 0.26158768])
        ])
        
        # 加载完整训练集
        full_dataset = CIFAR10_truncated(self.data_dir, train=True, download=True, transform=transform)
        
        return self._create_balanced_distillation_split(full_dataset, avoid_test_indices, num_classes=10)
    
    def _create_fashion_mnist_distillation_dataset(self, avoid_test_indices):
        """创建Fashion-MNIST蒸馏数据集"""
        from api.data_preprocessing.fashion_mnist.datasets import FashionMNIST_truncated
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530])
        ])
        
        # 加载完整训练集
        full_dataset = FashionMNIST_truncated(self.data_dir, train=True, download=True, transform=transform)
        
        return self._create_balanced_distillation_split(full_dataset, avoid_test_indices, num_classes=10)
    
    def _create_balanced_distillation_split(self, full_dataset, avoid_test_indices, num_classes):
        """
        创建平衡的蒸馏数据集划分
        
        Args:
            full_dataset: 完整的训练数据集
            avoid_test_indices: 需要避免的索引
            num_classes: 类别数量
        """
        # 获取所有可用的索引
        total_indices = list(range(len(full_dataset)))
        
        # 排除测试集索引
        if avoid_test_indices is not None:
            available_indices = [idx for idx in total_indices if idx not in avoid_test_indices]
        else:
            available_indices = total_indices
        
        # 按类别组织索引
        class_indices = defaultdict(list)
        for idx in available_indices:
            _, label = full_dataset[idx]
            class_indices[label].append(idx)
        
        # 计算每个类别的蒸馏样本数量
        distillation_indices = []
        validation_indices = []
        
        for class_id in range(num_classes):
            class_idx_list = class_indices[class_id]
            class_size = len(class_idx_list)
            
            # 计算该类别的蒸馏样本数
            num_distillation_samples = int(class_size * self.distillation_ratio)
            num_validation_samples = max(1, num_distillation_samples // 5)  # 20%用于验证
            num_distillation_samples = max(1, num_distillation_samples - num_validation_samples)
            
            # 随机选择样本
            np.random.shuffle(class_idx_list)
            
            distillation_indices.extend(class_idx_list[:num_distillation_samples])
            validation_indices.extend(class_idx_list[num_distillation_samples:num_distillation_samples + num_validation_samples])
        
        # 打乱索引
        np.random.shuffle(distillation_indices)
        np.random.shuffle(validation_indices)
        
        # 创建数据加载器
        distillation_subset = Subset(full_dataset, distillation_indices)
        validation_subset = Subset(full_dataset, validation_indices)
        
        self.distillation_loader = DataLoader(
            distillation_subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=False
        )
        
        self.validation_loader = DataLoader(
            validation_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        
        # 统计信息
        self.logger.info(f"蒸馏数据集创建完成:")
        self.logger.info(f"  蒸馏样本数: {len(distillation_indices)}")
        self.logger.info(f"  验证样本数: {len(validation_indices)}")
        self.logger.info(f"  蒸馏比例: {len(distillation_indices)/len(available_indices):.1%}")
        
        # 验证类别分布
        self._validate_class_distribution(distillation_indices, validation_indices, full_dataset, num_classes)
        
        return self.distillation_loader, self.validation_loader
    
    def _validate_class_distribution(self, distillation_indices, validation_indices, full_dataset, num_classes):
        """验证蒸馏数据集的类别分布是否均匀"""
        
        def count_class_distribution(indices, dataset):
            class_counts = np.zeros(num_classes, dtype=int)
            for idx in indices:
                _, label = dataset[idx]
                class_counts[label] += 1
            return class_counts
        
        distillation_counts = count_class_distribution(distillation_indices, full_dataset)
        validation_counts = count_class_distribution(validation_indices, full_dataset)
        
        self.logger.info("蒸馏数据集类别分布:")
        for class_id in range(num_classes):
            distill_ratio = distillation_counts[class_id] / len(distillation_indices)
            valid_ratio = validation_counts[class_id] / max(1, len(validation_indices))
            self.logger.info(f"  类别 {class_id}: 蒸馏集 {distillation_counts[class_id]} ({distill_ratio:.1%}), "
                           f"验证集 {validation_counts[class_id]} ({valid_ratio:.1%})")
        
        # 检查分布均匀性
        expected_ratio = 1.0 / num_classes
        max_deviation = max(abs(distillation_counts[i]/len(distillation_indices) - expected_ratio) 
                           for i in range(num_classes))
        
        if max_deviation > 0.05:  # 5%的偏差阈值
            self.logger.warning(f"蒸馏数据集类别分布不够均匀，最大偏差: {max_deviation:.1%}")
        else:
            self.logger.info("蒸馏数据集类别分布均匀")
    
    def get_distillation_loader(self):
        """获取蒸馏数据加载器"""
        if self.distillation_loader is None:
            raise ValueError("蒸馏数据集尚未创建，请先调用create_distillation_dataset()")
        return self.distillation_loader
    
    def get_validation_loader(self):
        """获取验证数据加载器"""
        if self.validation_loader is None:
            raise ValueError("蒸馏数据集尚未创建，请先调用create_distillation_dataset()")
        return self.validation_loader
    
    def get_data_statistics(self):
        """获取蒸馏数据集统计信息"""
        if self.distillation_loader is None:
            return None
        
        return {
            'distillation_samples': len(self.distillation_loader.dataset),
            'validation_samples': len(self.validation_loader.dataset),
            'distillation_batches': len(self.distillation_loader),
            'validation_batches': len(self.validation_loader),
            'batch_size': self.batch_size
        }
    
    def create_progressive_distillation_loader(self, difficulty_order='easy_to_hard'):
        """
        创建渐进式蒸馏数据加载器（从简单样本到复杂样本）
        
        Args:
            difficulty_order: 'easy_to_hard' 或 'hard_to_easy'
        """
        if self.distillation_loader is None:
            raise ValueError("蒸馏数据集尚未创建")
        
        # 这里可以基于样本的损失值或其他复杂度指标来排序
        # 简单实现：基于类别标签的顺序作为代理复杂度
        self.logger.info(f"创建{difficulty_order}渐进式蒸馏数据加载器...")
        
        return self.distillation_loader  # 暂时返回原始加载器