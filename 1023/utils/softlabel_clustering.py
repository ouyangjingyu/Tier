"""
客户端聚类管理器 - 基于软标签的聚类方案
最佳方案：本地测试集 + 软标签 + 分类器权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class SoftLabelClusterManager:
    """
    基于软标签的客户端聚类管理器
    
    核心流程：
    1. 预热训练客户端模型
    2. 在本地测试集上收集软标签
    3. 提取分类器为中心的特征（软标签统计 + 分类器权重范数）
    4. 基于特征相似度进行聚类
    """
    
    def __init__(self, num_classes=10, device='cuda'):
        """
        初始化聚类管理器
        
        Args:
            num_classes: 分类类别数
            device: 训练设备
        """
        self.num_classes = num_classes
        self.device = device
        self.client_soft_labels = {}  # {client_id: soft_labels_array}
        self.client_statistics = {}   # {client_id: feature_dict}
        self.cluster_assignments = {}  # {client_id: cluster_id}
        self.cluster_info = {}        # {cluster_id: cluster_info_dict}
        self.logger = logging.getLogger("SoftLabelClusterManager")
    
    def warmup_train_clients(self, client_models, train_data_local_dict, 
                            test_data_local_dict, warmup_epochs=15,
                            lr=0.01, early_stop_patience=5):
        """
        预热训练客户端模型，确保模型充分拟合本地数据
        
        Args:
            client_models: 客户端模型字典 {client_id: model}
            train_data_local_dict: 训练数据加载器字典
            test_data_local_dict: 测试数据加载器字典
            warmup_epochs: 预热训练轮数
            lr: 学习率
            early_stop_patience: 早停耐心值
            
        Returns:
            训练后的模型字典
        """
        self.logger.info(f"开始预热训练，轮数: {warmup_epochs}")
        
        for client_id, model in client_models.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"预热训练客户端 {client_id}")
            self.logger.info(f"{'='*60}")
            
            model = model.to(self.device)
            model.train()
            
            # 创建优化器（训练所有参数，包括共享层）
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=1e-4
            )
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=warmup_epochs
            )
            
            # 训练监控
            best_train_acc = 0.0
            patience_counter = 0
            train_history = []
            test_history = []
            
            # 训练循环
            for epoch in range(warmup_epochs):
                # 训练阶段
                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                batch_count = 0
                
                for data, target in train_data_local_dict[client_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播（使用本地分类器）
                    local_logits, _, _ = model(data)
                    loss = F.cross_entropy(local_logits, target)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    optimizer.step()
                    
                    # 统计
                    epoch_loss += loss.item()
                    _, pred = local_logits.max(1)
                    epoch_correct += pred.eq(target).sum().item()
                    epoch_total += target.size(0)
                    batch_count += 1
                
                # 计算训练准确率
                train_acc = 100.0 * epoch_correct / epoch_total
                avg_loss = epoch_loss / batch_count
                train_history.append(train_acc)
                
                # 测试阶段
                test_acc = self._evaluate_model(model, test_data_local_dict[client_id])
                test_history.append(test_acc)
                
                # 更新学习率
                scheduler.step()
                
                # 输出进度
                self.logger.info(
                    f"  Epoch {epoch+1:2d}/{warmup_epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                
                # Early stopping检查
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience and epoch >= 10:
                    self.logger.info(
                        f"  早停于 Epoch {epoch+1}，"
                        f"最佳训练准确率: {best_train_acc:.2f}%"
                    )
                    break
            
            # 训练完成总结
            self.logger.info(f"\n客户端 {client_id} 预热训练完成:")
            self.logger.info(f"  最佳训练准确率: {max(train_history):.2f}%")
            self.logger.info(f"  最终测试准确率: {test_history[-1]:.2f}%")
            self.logger.info(f"  实际训练轮数: {len(train_history)}")
            
            # 收敛性检查
            if max(train_history) < 30.0:
                self.logger.warning(
                    f"  ⚠️ 警告：客户端 {client_id} 训练准确率过低，"
                    f"可能未充分拟合！"
                )
            
            # 保存预热后的模型
            client_models[client_id] = model.cpu()
        
        self.logger.info("\n预热训练完成\n")
        return client_models
    
    def _evaluate_model(self, model, test_loader):
        """
        评估模型在测试集上的准确率
        
        Args:
            model: 待评估的模型
            test_loader: 测试数据加载器
            
        Returns:
            准确率（百分比）
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                local_logits, _, _ = model(data)
                _, pred = local_logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        model.train()
        return 100.0 * correct / total
    
    def collect_soft_labels_from_local_testset(self, client_models, 
                                               test_data_local_dict):
        """
        在客户端本地测试集上收集软标签
        
        这是最佳方案的关键步骤：使用本地测试集而非独立蒸馏数据集
        
        Args:
            client_models: 预热后的客户端模型字典
            test_data_local_dict: 本地测试数据字典
            
        Returns:
            软标签字典 {client_id: soft_labels_array}
        """
        self.logger.info("在本地测试集上收集客户端软标签...")
        
        for client_id, model in client_models.items():
            model = model.to(self.device)
            model.eval()
            
            all_soft_labels = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_data_local_dict[client_id]:
                    data = data.to(self.device)
                    
                    # 使用本地分类器生成软标签
                    local_logits, _, _ = model(data)
                    soft_labels = F.softmax(local_logits, dim=1)
                    
                    all_soft_labels.append(soft_labels.cpu().numpy())
                    all_targets.append(target.numpy())
            
            # 合并所有批次
            client_soft_labels = np.vstack(all_soft_labels)
            client_targets = np.concatenate(all_targets)
            
            self.client_soft_labels[client_id] = client_soft_labels
            
            # 分析软标签与真实标签的一致性
            predictions = np.argmax(client_soft_labels, axis=1)
            accuracy = np.mean(predictions == client_targets)
            
            self.logger.info(
                f"客户端 {client_id} - "
                f"测试集大小: {len(client_soft_labels)}, "
                f"预测准确率: {accuracy:.2%}"
            )
            
            # 警告：如果准确率太低，说明模型没学好
            if accuracy < 0.4:
                self.logger.warning(
                    f"⚠️ 客户端 {client_id} 在本地测试集上准确率过低 "
                    f"({accuracy:.2%})，可能需要增加预热训练轮数！"
                )
            
            model.cpu()
        
        return self.client_soft_labels
    
    def extract_classifier_focused_features(self, client_models):
        """
        提取分类器为中心的特征（最佳方案）
        
        特征组成（30维）：
        1. 软标签预测分布（10维）- 反映模型的预测偏好
        2. 分类器权重范数（10维）- 反映类别权重，关键特征
        3. 软标签置信度（10维）- 反映模型的确定性
        
        Args:
            client_models: 客户端模型字典
            
        Returns:
            统计特征字典 {client_id: feature_dict}
        """
        self.logger.info("提取分类器为中心的特征...")
        
        for client_id, soft_labels in self.client_soft_labels.items():
            # ===== 第1部分：软标签预测分布（10维） =====
            predictions = np.argmax(soft_labels, axis=1)
            predicted_distribution = np.bincount(
                predictions, minlength=self.num_classes
            ) / len(soft_labels)
            
            # ===== 第2部分：软标签平均置信度（10维） =====
            mean_confidence = np.mean(soft_labels, axis=0)
            
            # ===== 第3部分：分类器权重范数（10维）[关键] =====
            classifier_norms = self._extract_classifier_features(
                client_models[client_id]
            )
            
            # ===== 组合特征（30维） =====
            features = np.concatenate([
                predicted_distribution,   # 10维：预测偏好
                classifier_norms,         # 10维：类别权重（关键）
                mean_confidence          # 10维：置信度
            ])
            
            self.client_statistics[client_id] = {
                'features': features,
                'predicted_distribution': predicted_distribution,
                'classifier_norms': classifier_norms,
                'mean_confidence': mean_confidence
            }
            
            self.logger.info(
                f"客户端 {client_id} - 特征维度: {len(features)}"
            )
        
        return self.client_statistics
    
    def _extract_classifier_features(self, model):
        """
        提取分类器权重的L2范数作为类别偏好特征
        
        核心思想：分类器权重的范数模式直接反映模型对各类别的重视程度
        
        Args:
            model: 客户端模型
            
        Returns:
            类别权重范数向量（10维）
        """
        classifier_weight = None
        
        # 找到分类器的权重矩阵
        for name, param in model.named_parameters():
            if 'local_classifier' in name and 'weight' in name:
                # 应该是形状 [num_classes, feature_dim]
                if len(param.shape) == 2 and param.shape[0] == self.num_classes:
                    classifier_weight = param.data.cpu().numpy()
                    break
        
        if classifier_weight is None:
            self.logger.warning("未找到分类器权重，使用零向量")
            return np.zeros(self.num_classes)
        
        # 计算每个类别权重向量的L2范数
        class_norms = np.linalg.norm(classifier_weight, axis=1)
        
        return class_norms
    
    def cluster_clients(self, num_clusters=3, method='cosine_similarity'):
        """
        基于提取的特征对客户端进行聚类
        
        Args:
            num_clusters: 目标聚类数量
            method: 相似度度量方法 ('cosine_similarity' 或 'euclidean')
            
        Returns:
            聚类分配字典 {client_id: cluster_id}
        """
        self.logger.info(f"开始客户端聚类，目标组数: {num_clusters}")
        
        if not self.client_statistics:
            raise ValueError("请先提取统计特征")
        
        # 构建特征矩阵
        client_ids = sorted(self.client_statistics.keys())
        feature_matrix = np.array([
            self.client_statistics[cid]['features'] 
            for cid in client_ids
        ])
        
        self.logger.info(f"特征矩阵形状: {feature_matrix.shape}")
        
        # 执行聚类
        if method == 'cosine_similarity':
            # 使用余弦相似度
            similarity_matrix = cosine_similarity(feature_matrix)
            distance_matrix = 1 - similarity_matrix
            
            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        else:
            # 使用欧氏距离
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric='euclidean',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(feature_matrix)
        
        # 保存聚类结果
        for i, client_id in enumerate(client_ids):
            self.cluster_assignments[client_id] = cluster_labels[i]
        
        # 分析聚类结果
        self._analyze_clusters()
        
        self.logger.info(f"聚类完成，分配结果: {self.cluster_assignments}")
        
        return self.cluster_assignments
    
    def _analyze_clusters(self):
        """分析并记录聚类结果的统计信息"""
        self.logger.info("\n分析聚类结果...")
        
        # 按组组织客户端
        groups = defaultdict(list)
        for client_id, cluster_id in self.cluster_assignments.items():
            groups[cluster_id].append(client_id)
        
        # 分析每个组
        for cluster_id, client_list in groups.items():
            self.logger.info(f"\n组 {cluster_id}: 客户端 {client_list}")
            
            # 收集组内特征
            group_features = []
            group_predicted_dists = []
            group_classifier_norms = []
            
            for client_id in client_list:
                stats = self.client_statistics[client_id]
                group_features.append(stats['features'])
                group_predicted_dists.append(stats['predicted_distribution'])
                group_classifier_norms.append(stats['classifier_norms'])
            
            group_features = np.array(group_features)
            group_predicted_dists = np.array(group_predicted_dists)
            group_classifier_norms = np.array(group_classifier_norms)
            
            # 计算组内相似度
            if len(group_features) > 1:
                sim_matrix = cosine_similarity(group_features)
                # 上三角的平均值（排除对角线）
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                avg_similarity = sim_matrix[mask].mean()
            else:
                avg_similarity = 1.0
            
            # 计算平均预测分布
            avg_predicted_dist = np.mean(group_predicted_dists, axis=0)
            
            # 计算平均分类器范数
            avg_classifier_norms = np.mean(group_classifier_norms, axis=0)
            
            # 找出主要类别（预测分布 > 15%）
            main_classes = np.where(avg_predicted_dist > 0.15)[0]
            
            # 找出权重最大的类别（分类器范数）
            top_weight_classes = np.argsort(avg_classifier_norms)[-3:][::-1]
            
            # 保存聚类信息
            self.cluster_info[cluster_id] = {
                'clients': client_list,
                'size': len(client_list),
                'avg_similarity': avg_similarity,
                'avg_predicted_distribution': avg_predicted_dist,
                'avg_classifier_norms': avg_classifier_norms,
                'main_classes': main_classes.tolist(),
                'top_weight_classes': top_weight_classes.tolist(),
                'total_samples': sum([
                    len(self.client_soft_labels[cid]) 
                    for cid in client_list
                ])
            }
            
            self.logger.info(f"  组内相似度: {avg_similarity:.3f}")
            self.logger.info(f"  主要预测类别: {main_classes.tolist()}")
            self.logger.info(f"  权重最大类别: {top_weight_classes.tolist()}")
            self.logger.info(f"  平均预测分布: {avg_predicted_dist.round(3)}")
    
    def get_cluster_assignments(self):
        """获取聚类分配结果"""
        return self.cluster_assignments
    
    def get_cluster_info(self):
        """获取聚类信息"""
        return self.cluster_info
    
    def get_clients_in_cluster(self, cluster_id):
        """获取指定组中的客户端列表"""
        return [
            client_id for client_id, cid in self.cluster_assignments.items() 
            if cid == cluster_id
        ]