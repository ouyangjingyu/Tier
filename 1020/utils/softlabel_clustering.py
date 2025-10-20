import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import copy

class SoftLabelClusterManager:
    """基于软标签的客户端聚类管理器"""
    
    def __init__(self, num_classes=10, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.client_soft_labels = {}
        self.client_statistics = {}
        self.cluster_assignments = {}
        self.cluster_info = {}
        self.logger = logging.getLogger("SoftLabelClusterManager")
        
    def warmup_train_clients(self, client_models, train_data_local_dict, 
                        test_data_local_dict,  # 新增：测试集
                        warmup_epochs=15,  # 增加到15轮
                        lr=0.01,
                        early_stop_patience=5):
        """
        增强版预热训练：确保模型充分拟合本地数据
        """
        self.logger.info(f"开始增强预热训练，轮数: {warmup_epochs}")
        
        for client_id, model in client_models.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"预热训练客户端 {client_id}")
            self.logger.info(f"{'='*50}")
            
            model = model.to(self.device)
            model.train()
            
            # 创建优化器 - 训练所有参数（包括共享层）
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
            
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
                    
                    # 前向传播 - 使用本地分类器
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
                    self.logger.info(f"  早停于 Epoch {epoch+1}，最佳训练准确率: {best_train_acc:.2f}%")
                    break
            
            # 训练完成总结
            self.logger.info(f"\n客户端 {client_id} 预热训练完成:")
            self.logger.info(f"  最佳训练准确率: {max(train_history):.2f}%")
            self.logger.info(f"  最终测试准确率: {test_history[-1]:.2f}%")
            self.logger.info(f"  训练轮数: {len(train_history)}")
            
            # 收敛性检查
            if max(train_history) < 30.0:
                self.logger.warning(f"  ⚠️ 警告：客户端 {client_id} 训练准确率过低，可能未充分拟合！")
            
            # 保存预热后的模型
            client_models[client_id] = model.cpu()
        
        self.logger.info("\n预热训练完成")
        return client_models

    def _evaluate_model(self, model, test_loader):
        """评估模型在测试集上的准确率"""
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

    def collect_soft_labels(self, client_models, distillation_loader):
        """
        在蒸馏数据集上收集客户端的软标签
        
        Args:
            client_models: 预热后的客户端模型字典
            distillation_loader: 蒸馏数据加载器
        
        Returns:
            client_soft_labels: {client_id: soft_labels_array}
        """
        self.logger.info("开始收集客户端软标签...")
        
        # 将蒸馏数据集转换为tensor（一次性加载）
        all_data = []
        all_targets = []
        
        for data, target in distillation_loader:
            all_data.append(data)
            all_targets.append(target)
        
        distill_data = torch.cat(all_data, dim=0)
        distill_targets = torch.cat(all_targets, dim=0)
        
        self.logger.info(f"蒸馏数据集大小: {distill_data.size(0)} 样本")
        
        # 对每个客户端收集软标签
        for client_id, model in client_models.items():
            model = model.to(self.device)
            model.eval()
            
            all_soft_labels = []
            
            with torch.no_grad():
                # 分批处理避免内存溢出
                batch_size = 128
                for i in range(0, len(distill_data), batch_size):
                    batch_data = distill_data[i:i+batch_size].to(self.device)
                    
                    # 使用本地分类器生成软标签
                    local_logits, _, _ = model(batch_data)
                    soft_labels = F.softmax(local_logits, dim=1)
                    
                    all_soft_labels.append(soft_labels.cpu().numpy())
            
            # 合并所有批次
            client_soft_labels = np.vstack(all_soft_labels)
            self.client_soft_labels[client_id] = client_soft_labels
            
            self.logger.info(f"客户端 {client_id} 软标签收集完成，"
                           f"形状: {client_soft_labels.shape}")
            
            model.cpu()
        
        return self.client_soft_labels
    
    def extract_statistical_features(self):
        """
        从软标签中提取统计特征
        
        使用三种统计摘要：
        1. 预测分布（各类别被预测为top-1的比例）
        2. 平均置信度（各类别的平均最大概率）
        3. 熵特征（预测的不确定性）
        """
        self.logger.info("提取软标签统计特征...")
        
        for client_id, soft_labels in self.client_soft_labels.items():
            # 方案A: 类别置信度分布
            mean_confidence = np.mean(soft_labels, axis=0)  # 10维
            max_confidence = np.max(soft_labels, axis=0)    # 10维
            std_confidence = np.std(soft_labels, axis=0)    # 10维
            
            # 方案B: 预测偏好向量（推荐）
            predictions = np.argmax(soft_labels, axis=1)
            predicted_counts = np.bincount(predictions, minlength=self.num_classes)
            predicted_distribution = predicted_counts / len(soft_labels)  # 10维
            
            # 每个类别被预测时的平均最大置信度
            avg_max_conf_per_class = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                mask = predictions == c
                if mask.sum() > 0:
                    max_probs = np.max(soft_labels[mask], axis=1)
                    avg_max_conf_per_class[c] = np.mean(max_probs)
            
            # 方案C: 熵特征
            # 样本级熵
            sample_entropy = -np.sum(soft_labels * np.log(soft_labels + 1e-10), axis=1)
            mean_entropy = np.mean(sample_entropy)
            std_entropy = np.std(sample_entropy)
            
            # 高置信度样本比例
            max_probs = np.max(soft_labels, axis=1)
            high_conf_ratio = np.mean(max_probs > 0.8)
            
            # 组合所有特征
            statistical_features = np.concatenate([
                predicted_distribution,      # 10维：预测分布
                avg_max_conf_per_class,      # 10维：平均置信度
                mean_confidence,             # 10维：均值置信度
                [mean_entropy, std_entropy, high_conf_ratio]  # 3维：熵特征
            ])
            
            self.client_statistics[client_id] = {
                'features': statistical_features,
                'predicted_distribution': predicted_distribution,
                'mean_confidence': mean_confidence,
                'mean_entropy': mean_entropy,
                'high_conf_ratio': high_conf_ratio
            }
            
            self.logger.info(f"客户端 {client_id} 统计特征: "
                           f"特征维度={len(statistical_features)}, "
                           f"预测熵={mean_entropy:.3f}, "
                           f"高置信度比例={high_conf_ratio:.2%}")
        
        return self.client_statistics
    
    def cluster_clients(self, num_clusters=3, method='cosine_similarity'):
        """
        基于统计特征对客户端进行聚类
        
        Args:
            num_clusters: 聚类数量
            method: 相似度度量方法
        """
        self.logger.info(f"开始客户端聚类，目标组数: {num_clusters}")
        
        if not self.client_statistics:
            raise ValueError("请先提取统计特征")
        
        # 构建特征矩阵
        client_ids = list(self.client_statistics.keys())
        feature_matrix = np.array([
            self.client_statistics[cid]['features'] 
            for cid in client_ids
        ])
        
        self.logger.info(f"特征矩阵形状: {feature_matrix.shape}")
        
        # 执行聚类
        if method == 'cosine_similarity':
            # 计算余弦相似度矩阵
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
            # 直接使用欧氏距离
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
        """分析聚类结果"""
        self.logger.info("分析聚类结果...")
        
        # 按组组织客户端
        groups = defaultdict(list)
        for client_id, cluster_id in self.cluster_assignments.items():
            groups[cluster_id].append(client_id)
        
        # 分析每个组
        for cluster_id, client_list in groups.items():
            self.logger.info(f"\n组 {cluster_id}: 客户端 {client_list}")
            
            # 计算组内平均统计量
            group_features = []
            group_predicted_dists = []
            
            for client_id in client_list:
                stats = self.client_statistics[client_id]
                group_features.append(stats['features'])
                group_predicted_dists.append(stats['predicted_distribution'])
            
            group_features = np.array(group_features)
            group_predicted_dists = np.array(group_predicted_dists)
            
            # 计算组内相似度
            if len(group_features) > 1:
                sim_matrix = cosine_similarity(group_features)
                # 上三角的平均值
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                avg_similarity = sim_matrix[mask].mean()
            else:
                avg_similarity = 1.0
            
            # 计算平均预测分布
            avg_predicted_dist = np.mean(group_predicted_dists, axis=0)
            main_classes = np.where(avg_predicted_dist > 0.15)[0]
            
            self.cluster_info[cluster_id] = {
                'clients': client_list,
                'size': len(client_list),
                'avg_similarity': avg_similarity,
                'avg_predicted_distribution': avg_predicted_dist,
                'main_classes': main_classes.tolist()
            }
            
            self.logger.info(f"  组内相似度: {avg_similarity:.3f}")
            self.logger.info(f"  主要预测类别: {main_classes.tolist()}")
            self.logger.info(f"  平均预测分布: {avg_predicted_dist.round(3)}")
    
    def get_cluster_assignments(self):
        """获取聚类分配结果"""
        return self.cluster_assignments
    
    def get_cluster_info(self):
        """获取聚类信息"""
        return self.cluster_info
    
    def get_clients_in_cluster(self, cluster_id):
        """获取指定组中的客户端列表"""
        return [client_id for client_id, cid in self.cluster_assignments.items() 
                if cid == cluster_id]

    def collect_soft_labels_from_local_testset(self, client_models, test_data_local_dict):
        """
        在客户端本地测试集上收集软标签（更直接反映本地分布）
        
        Args:
            client_models: 预热后的客户端模型字典
            test_data_local_dict: 本地测试数据字典
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
                    f"⚠️ 客户端 {client_id} 在本地测试集上准确率过低 ({accuracy:.2%})，"
                    f"可能需要增加预热训练轮数！"
                )
            
            model.cpu()
        
        return self.client_soft_labels

    def extract_hybrid_features(self, client_models):
        """
        混合特征提取：结合软标签和模型参数
        """
        self.logger.info("提取混合统计特征...")
        
        for client_id, soft_labels in self.client_soft_labels.items():
            # 方案1: 软标签统计特征（已有）
            predictions = np.argmax(soft_labels, axis=1)
            predicted_distribution = np.bincount(predictions, minlength=self.num_classes) / len(soft_labels)
            
            mean_confidence = np.mean(soft_labels, axis=0)
            
            avg_max_conf_per_class = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                mask = predictions == c
                if mask.sum() > 0:
                    max_probs = np.max(soft_labels[mask], axis=1)
                    avg_max_conf_per_class[c] = np.mean(max_probs)
            
            sample_entropy = -np.sum(soft_labels * np.log(soft_labels + 1e-10), axis=1)
            mean_entropy = np.mean(sample_entropy)
            std_entropy = np.std(sample_entropy)
            
            max_probs = np.max(soft_labels, axis=1)
            high_conf_ratio = np.mean(max_probs > 0.8)
            
            # 方案2: 个性化层参数的统计特征（新增）
            model = client_models[client_id]
            param_features = self._extract_parameter_statistics(model)
            
            # 组合所有特征
            statistical_features = np.concatenate([
                predicted_distribution,      # 10维
                avg_max_conf_per_class,      # 10维
                mean_confidence,             # 10维
                [mean_entropy, std_entropy, high_conf_ratio],  # 3维
                param_features               # N维（参数统计）
            ])
            
            self.client_statistics[client_id] = {
                'features': statistical_features,
                'predicted_distribution': predicted_distribution,
                'mean_confidence': mean_confidence,
                'mean_entropy': mean_entropy,
                'high_conf_ratio': high_conf_ratio,
                'param_features': param_features
            }
            
            self.logger.info(
                f"客户端 {client_id} - "
                f"特征维度: {len(statistical_features)}, "
                f"预测熵: {mean_entropy:.3f}"
            )
        
        return self.client_statistics

    def _extract_parameter_statistics(self, model):
        """
        从个性化层提取参数统计特征
        """
        param_stats = []
        
        for name, param in model.named_parameters():
            # 只提取个性化层的参数
            if 'personalized_path' in name or 'local_classifier' in name:
                param_data = param.data.cpu().numpy().flatten()
                
                # 提取统计量
                param_stats.extend([
                    np.mean(param_data),
                    np.std(param_data),
                    np.max(np.abs(param_data))
                ])
        
        return np.array(param_stats)

    def extract_improved_hybrid_features(self, client_models):
        """
        改进的混合特征提取：重点关注分类器权重
        
        特征组成：
        1. 软标签统计（33维）- 反映模型行为
        2. 分类器权重范数（10维）- 反映类别偏好 [关键]
        3. 分类器权重方向（可选）- 反映决策模式
        4. 特征层统计（少量）- 辅助信息
        """
        self.logger.info("提取改进的混合统计特征（重点：分类器权重）...")
        
        for client_id, soft_labels in self.client_soft_labels.items():
            # ========== 第1部分：软标签统计（33维）==========
            predictions = np.argmax(soft_labels, axis=1)
            predicted_distribution = np.bincount(
                predictions, minlength=self.num_classes
            ) / len(soft_labels)
            
            mean_confidence = np.mean(soft_labels, axis=0)
            
            avg_max_conf_per_class = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                mask = predictions == c
                if mask.sum() > 0:
                    max_probs = np.max(soft_labels[mask], axis=1)
                    avg_max_conf_per_class[c] = np.mean(max_probs)
            
            sample_entropy = -np.sum(soft_labels * np.log(soft_labels + 1e-10), axis=1)
            mean_entropy = np.mean(sample_entropy)
            std_entropy = np.std(sample_entropy)
            
            max_probs = np.max(soft_labels, axis=1)
            high_conf_ratio = np.mean(max_probs > 0.8)
            
            # ========== 第2部分：分类器权重特征（10-20维）[关键] ==========
            classifier_features = self._extract_classifier_features(
                client_models[client_id]
            )
            
            # ========== 第3部分：特征层统计（3-5维）[辅助] ==========
            feature_layer_features = self._extract_lightweight_feature_stats(
                client_models[client_id]
            )
            
            # ========== 组合特征（权重配比很重要！）==========
            # 方案1: 平衡组合
            statistical_features = np.concatenate([
                predicted_distribution,      # 10维
                avg_max_conf_per_class,      # 10维
                mean_confidence,             # 10维
                [mean_entropy, std_entropy, high_conf_ratio],  # 3维
                classifier_features,         # 10-20维 [重点]
                feature_layer_features       # 3-5维
            ])
            
            # 方案2: 加权组合（可选，通过缩放强调分类器）
            # weighted_features = np.concatenate([
            #     predicted_distribution * 1.0,
            #     classifier_features * 2.0,      # 2x权重
            #     feature_layer_features * 0.5    # 0.5x权重
            # ])
            
            self.client_statistics[client_id] = {
                'features': statistical_features,
                'predicted_distribution': predicted_distribution,
                'classifier_features': classifier_features,
                'feature_layer_features': feature_layer_features,
                'mean_entropy': mean_entropy,
                'high_conf_ratio': high_conf_ratio
            }
            
            self.logger.info(
                f"客户端 {client_id} - "
                f"总特征维度: {len(statistical_features)}, "
                f"分类器特征: {len(classifier_features)}维"
            )
        
        return self.client_statistics

    def _extract_classifier_features(self, model):
        """
        提取分类器的关键特征
        
        核心思想：分类器权重的范数模式直接反映类别偏好
        """
        classifier_weight = None
        
        # 找到分类器的权重矩阵
        for name, param in model.named_parameters():
            if 'local_classifier' in name and 'weight' in name:
                if len(param.shape) == 2 and param.shape[0] == self.num_classes:
                    classifier_weight = param.data.cpu().numpy()
                    break
        
        if classifier_weight is None:
            self.logger.warning("未找到分类器权重，使用零向量")
            return np.zeros(self.num_classes)
        
        # 方案A: 每个类别权重向量的L2范数（10维）[最简单最有效]
        class_norms = np.linalg.norm(classifier_weight, axis=1)
        
        # 方案B: 归一化范数（使其成为分布）
        # class_norms_normalized = class_norms / (np.sum(class_norms) + 1e-8)
        
        # 方案C: 加上权重的方向信息（20维）
        # top_weight_indices = np.argsort(class_norms)[-3:]  # 前3个主导类
        # direction_features = classifier_weight[top_weight_indices].flatten()[:10]
        # return np.concatenate([class_norms, direction_features])
        
        return class_norms  # 返回10维范数向量

    def _extract_lightweight_feature_stats(self, model):
        """
        轻量级特征层统计（只提取关键统计量）
        """
        all_params = []
        
        for name, param in model.named_parameters():
            if 'personalized_path' in name:
                param_data = param.data.cpu().numpy().flatten()
                all_params.append(param_data)
        
        if not all_params:
            return np.array([0.0, 0.0, 0.0])
        
        all_params = np.concatenate(all_params)
        
        # 只提取3个关键统计量
        return np.array([
            np.mean(np.abs(all_params)),  # 绝对值均值
            np.std(all_params),            # 标准差
            np.max(np.abs(all_params))     # 最大绝对值
        ])

    def extract_classifier_focused_features(self, client_models):
        """
        分类器为中心的特征提取（简化版）
        
        特征组成：
        - 软标签预测分布（10维）
        - 分类器权重范数（10维）[关键]
        - 软标签置信度（10维）
        """
        self.logger.info("提取分类器为中心的特征...")
        
        for client_id, soft_labels in self.client_soft_labels.items():
            # 软标签统计
            predictions = np.argmax(soft_labels, axis=1)
            predicted_distribution = np.bincount(
                predictions, minlength=self.num_classes
            ) / len(soft_labels)
            
            mean_confidence = np.mean(soft_labels, axis=0)
            
            # 分类器权重范数
            classifier_norms = self._extract_classifier_features(client_models[client_id])
            
            # 组合特征（30维）
            features = np.concatenate([
                predicted_distribution,   # 10维
                classifier_norms,         # 10维 [关键]
                mean_confidence          # 10维
            ])
            
            self.client_statistics[client_id] = {
                'features': features,
                'predicted_distribution': predicted_distribution,
                'classifier_norms': classifier_norms
            }
        
        return self.client_statistics


def compare_clustering_results(true_assignments, predicted_assignments):
    """
    比较两种聚类结果的一致性
    
    Args:
        true_assignments: 真实聚类（原方案）{client_id: cluster_id}
        predicted_assignments: 预测聚类（新方案）{client_id: cluster_id}
    
    Returns:
        metrics: 评估指标字典
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    # 确保客户端ID顺序一致
    client_ids = sorted(true_assignments.keys())
    
    true_labels = [true_assignments[cid] for cid in client_ids]
    pred_labels = [predicted_assignments[cid] for cid in client_ids]
    
    # 计算评估指标
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # 计算精确匹配率（需要找最佳标签映射）
    accuracy = _compute_clustering_accuracy(true_labels, pred_labels)
    
    metrics = {
        'NMI': nmi,  # 归一化互信息：0-1，越高越好
        'ARI': ari,  # 调整兰德指数：-1到1，越高越好
        'Accuracy': accuracy  # 精确匹配率
    }
    
    return metrics


def _compute_clustering_accuracy(true_labels, pred_labels):
    """
    计算聚类精确度（考虑标签映射）
    """
    from scipy.optimize import linear_sum_assignment
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # 构建混淆矩阵
    n_clusters = max(max(true_labels), max(pred_labels)) + 1
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    # 使用匈牙利算法找最佳映射
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # 计算精确度
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    
    return accuracy



