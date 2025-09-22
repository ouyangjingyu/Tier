import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class MultiTeacherDistillationLoss(nn.Module):
    """多教师知识蒸馏损失函数"""
    
    def __init__(self, temperature=4.0, alpha=0.3, beta=0.7):
        """
        Args:
            temperature: 软化温度参数
            alpha: 真实标签损失权重
            beta: 蒸馏损失权重
        """
        super(MultiTeacherDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        self.logger = logging.getLogger("MultiTeacherDistillationLoss")
        
    def forward(self, student_logits, teacher_logits_list, teacher_weights, true_labels):
        """
        计算多教师蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, num_classes]
            teacher_logits_list: 教师模型输出列表 [num_teachers, batch_size, num_classes]
            teacher_weights: 教师权重 [num_teachers]
            true_labels: 真实标签 [batch_size]
        
        Returns:
            总损失，分项损失字典
        """
        batch_size, num_classes = student_logits.shape
        device = student_logits.device
        
        # 1. 真实标签损失
        ce_loss = self.ce_loss(student_logits, true_labels)
        
        # 2. 蒸馏损失
        distillation_loss = torch.tensor(0.0, device=device)
        individual_kl_losses = []
        
        # 确保教师权重和为1
        teacher_weights = torch.tensor(teacher_weights, device=device)
        teacher_weights = teacher_weights / (teacher_weights.sum() + 1e-8)
        
        for i, teacher_logits in enumerate(teacher_logits_list):
            if teacher_logits is None:
                continue
                
            # 软化概率分布
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            student_soft_log = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # KL散度损失
            kl_loss = self.kl_loss(student_soft_log, teacher_soft)
            
            # 加权累加
            weighted_kl_loss = teacher_weights[i] * kl_loss
            distillation_loss += weighted_kl_loss
            
            individual_kl_losses.append(kl_loss.item())
        
        # 温度平方缩放
        distillation_loss *= (self.temperature ** 2)
        
        # 总损失
        total_loss = self.alpha * ce_loss + self.beta * distillation_loss
        
        # 返回详细信息
        loss_info = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'individual_kl_losses': individual_kl_losses,
            'teacher_weights': teacher_weights.cpu().numpy().tolist()
        }
        
        return total_loss, loss_info
    
    def update_temperature(self, new_temperature):
        """动态调整温度参数"""
        self.temperature = new_temperature
        self.logger.info(f"温度参数更新为: {new_temperature}")
    
    def update_loss_weights(self, alpha, beta):
        """更新损失权重"""
        self.alpha = alpha
        self.beta = beta
        self.logger.info(f"损失权重更新为: α={alpha}, β={beta}")

class AdaptiveTemperatureScheduler:
    """自适应温度调度器"""
    
    def __init__(self, initial_temp=4.0, min_temp=2.0, max_temp=6.0):
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.loss_history = []
        
    def step(self, current_loss):
        """根据损失历史调整温度"""
        self.loss_history.append(current_loss)
        
        # 保留最近10个损失值
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        
        if len(self.loss_history) >= 5:
            # 计算损失趋势
            recent_losses = self.loss_history[-5:]
            loss_trend = recent_losses[-1] - recent_losses[0]
            
            # 根据趋势调整温度
            if loss_trend > 0.1:  # 损失上升，增加温度（软化分布）
                self.current_temp = min(self.max_temp, self.current_temp + 0.2)
            elif loss_trend < -0.1:  # 损失下降，降低温度（锐化分布）
                self.current_temp = max(self.min_temp, self.current_temp - 0.2)
        
        return self.current_temp

class FeatureAlignmentLoss(nn.Module):
    """特征对齐损失，用于师生模型特征空间对齐"""
    
    def __init__(self, alignment_type='cosine'):
        super(FeatureAlignmentLoss, self).__init__()
        self.alignment_type = alignment_type
        
    def forward(self, student_features, teacher_features_list, teacher_weights):
        """
        计算特征对齐损失
        
        Args:
            student_features: 学生模型特征 [batch_size, feature_dim]
            teacher_features_list: 教师模型特征列表
            teacher_weights: 教师权重
        """
        if not teacher_features_list:
            return torch.tensor(0.0, device=student_features.device)
        
        alignment_loss = torch.tensor(0.0, device=student_features.device)
        
        # 确保特征是2D
        if len(student_features.shape) > 2:
            student_features = F.adaptive_avg_pool2d(student_features, (1, 1)).flatten(1)
        
        # 标准化学生特征
        student_features_norm = F.normalize(student_features, dim=1)
        
        for i, teacher_features in enumerate(teacher_features_list):
            if teacher_features is None:
                continue
                
            # 确保教师特征是2D
            if len(teacher_features.shape) > 2:
                teacher_features = F.adaptive_avg_pool2d(teacher_features, (1, 1)).flatten(1)
            
            # 标准化教师特征
            teacher_features_norm = F.normalize(teacher_features, dim=1)
            
            if self.alignment_type == 'cosine':
                # 余弦相似度对齐
                cosine_sim = torch.sum(student_features_norm * teacher_features_norm, dim=1)
                alignment_loss += teacher_weights[i] * (1 - cosine_sim).mean()
            elif self.alignment_type == 'mse':
                # MSE对齐
                mse_loss = F.mse_loss(student_features_norm, teacher_features_norm)
                alignment_loss += teacher_weights[i] * mse_loss
        
        return alignment_loss

class DynamicWeightCalculator:
    """动态权重计算器，用于计算教师模型权重"""
    
    def __init__(self, weight_strategy='consistency_based'):
        self.weight_strategy = weight_strategy
        self.performance_history = {}
        
    def calculate_teacher_weights(self, cluster_info, cluster_models=None, validation_loader=None, device='cuda'):
        """
        计算教师模型权重
        
        Args:
            cluster_info: 聚类信息
            cluster_models: 聚类模型（用于性能评估）
            validation_loader: 验证数据加载器
            device: 设备
        
        Returns:
            教师权重字典 {cluster_id: weight}
        """
        cluster_ids = list(cluster_info.keys())
        num_clusters = len(cluster_ids)
        
        if self.weight_strategy == 'uniform':
            # 均匀权重
            return {cluster_id: 1.0 / num_clusters for cluster_id in cluster_ids}
        
        elif self.weight_strategy == 'consistency_based':
            # 基于组内一致性的权重
            weights = {}
            total_consistency = 0.0
            
            for cluster_id, info in cluster_info.items():
                # 使用组内平均相似性作为权重
                consistency = info.get('avg_similarity', 0.5)
                weights[cluster_id] = consistency
                total_consistency += consistency
            
            # 归一化
            if total_consistency > 0:
                for cluster_id in weights:
                    weights[cluster_id] /= total_consistency
            else:
                # 回退到均匀权重
                for cluster_id in weights:
                    weights[cluster_id] = 1.0 / num_clusters
                    
            return weights
        
        elif self.weight_strategy == 'performance_based' and cluster_models is not None and validation_loader is not None:
            # 基于验证性能的权重
            return self._calculate_performance_based_weights(cluster_models, validation_loader, device)
        
        elif self.weight_strategy == 'data_size_based':
            # 基于数据量的权重
            weights = {}
            total_samples = sum(info['total_samples'] for info in cluster_info.values())
            
            for cluster_id, info in cluster_info.items():
                weights[cluster_id] = info['total_samples'] / total_samples
                
            return weights
        
        else:
            # 默认均匀权重
            return {cluster_id: 1.0 / num_clusters for cluster_id in cluster_ids}
    
    def _calculate_performance_based_weights(self, cluster_models, validation_loader, device):
        """基于验证集性能计算权重"""
        cluster_accuracies = {}
        
        for cluster_id, models in cluster_models.items():
            client_model, server_model = models
            
            # 评估模型性能
            accuracy = self._evaluate_model(client_model, server_model, validation_loader, device)
            cluster_accuracies[cluster_id] = accuracy
            
            # 更新历史性能
            if cluster_id not in self.performance_history:
                self.performance_history[cluster_id] = []
            self.performance_history[cluster_id].append(accuracy)
            
            # 保持最近5次的历史
            if len(self.performance_history[cluster_id]) > 5:
                self.performance_history[cluster_id].pop(0)
        
        # 基于性能计算权重（性能越好权重越大）
        total_accuracy = sum(cluster_accuracies.values())
        if total_accuracy > 0:
            weights = {cluster_id: acc / total_accuracy for cluster_id, acc in cluster_accuracies.items()}
        else:
            # 回退到均匀权重
            num_clusters = len(cluster_accuracies)
            weights = {cluster_id: 1.0 / num_clusters for cluster_id in cluster_accuracies.keys()}
        
        return weights
    
    def _evaluate_model(self, client_model, server_model, validation_loader, device):
        """评估单个模型的性能"""
        client_model.eval()
        server_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                _, shared_features, _ = client_model(data)
                logits = server_model(shared_features)
                
                # 计算准确率
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / max(1, total)