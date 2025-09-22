import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import numpy as np
from collections import defaultdict
import time

from .distillation_losses import MultiTeacherDistillationLoss, AdaptiveTemperatureScheduler, DynamicWeightCalculator
from .distillation_data_manager import DistillationDataManager

class StudentModelTrainer:
    """学生模型训练器"""
    
    def __init__(self, device='cuda', lr=0.001, weight_decay=1e-4):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.logger = logging.getLogger("StudentModelTrainer")
        
        # 损失函数和调度器
        self.distillation_loss = MultiTeacherDistillationLoss(temperature=4.0, alpha=0.3, beta=0.7)
        self.temp_scheduler = AdaptiveTemperatureScheduler()
        
    def train_student_model(self, student_client_model, student_server_model, 
                          teacher_models, teacher_weights,
                          distillation_loader, validation_loader,
                          num_epochs=10, early_stop_patience=3):
        """
        训练学生模型
        
        Args:
            student_client_model: 学生客户端模型
            student_server_model: 学生服务器模型
            teacher_models: 教师模型字典 {cluster_id: (client_model, server_model)}
            teacher_weights: 教师权重字典 {cluster_id: weight}
            distillation_loader: 蒸馏数据加载器
            validation_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stop_patience: 早停耐心值
        
        Returns:
            训练统计信息
        """
        self.logger.info("开始学生模型蒸馏训练...")
        
        # 设置训练模式
        student_client_model.train()
        student_server_model.train()
        
        # 设置教师模型为评估模式
        for cluster_id, (teacher_client, teacher_server) in teacher_models.items():
            teacher_client.eval()
            teacher_server.eval()
        
        # 创建优化器
        student_params = list(student_client_model.parameters()) + list(student_server_model.parameters())
        optimizer = optim.Adam(student_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
        
        # 训练统计
        train_stats = {
            'train_losses': [],
            'val_accuracies': [],
            'best_val_accuracy': 0.0,
            'best_epoch': 0,
            'teacher_weights_history': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, loss_details = self._train_one_epoch(
                student_client_model, student_server_model,
                teacher_models, teacher_weights,
                distillation_loader, optimizer
            )
            
            # 验证
            val_accuracy, val_loss = self._validate_student_model(
                student_client_model, student_server_model, validation_loader
            )
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 更新温度
            new_temp = self.temp_scheduler.step(train_loss)
            self.distillation_loss.update_temperature(new_temp)
            
            # 记录统计信息
            train_stats['train_losses'].append(train_loss)
            train_stats['val_accuracies'].append(val_accuracy)
            train_stats['teacher_weights_history'].append(copy.deepcopy(teacher_weights))
            
            # 早停检查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                train_stats['best_val_accuracy'] = best_val_accuracy
                train_stats['best_epoch'] = epoch
                patience_counter = 0
                
                # 保存最佳模型状态
                train_stats['best_student_client_state'] = copy.deepcopy(student_client_model.state_dict())
                train_stats['best_student_server_state'] = copy.deepcopy(student_server_model.state_dict())
            else:
                patience_counter += 1
            
            # 输出训练信息
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.2f}%, "
                           f"Temp: {new_temp:.2f}, "
                           f"Time: {epoch_time:.2f}s")
            
            # 早停
            if patience_counter >= early_stop_patience:
                self.logger.info(f"早停于epoch {epoch+1}，最佳验证准确率: {best_val_accuracy:.2f}%")
                break
        
        # 加载最佳模型
        if 'best_student_client_state' in train_stats:
            student_client_model.load_state_dict(train_stats['best_student_client_state'])
            student_server_model.load_state_dict(train_stats['best_student_server_state'])
            self.logger.info(f"已加载最佳模型 (epoch {train_stats['best_epoch']+1})")
        
        return train_stats
    
    def _train_one_epoch(self, student_client_model, student_server_model,
                        teacher_models, teacher_weights,
                        distillation_loader, optimizer):
        """训练一个epoch"""
        total_loss = 0.0
        loss_details_sum = defaultdict(float)
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(distillation_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 学生模型前向传播
            _, student_shared_features, _ = student_client_model(data)
            student_logits = student_server_model(student_shared_features)
            
            # 教师模型前向传播
            teacher_logits_list = []
            weights_list = []
            
            with torch.no_grad():
                for cluster_id, (teacher_client, teacher_server) in teacher_models.items():
                    teacher_client = teacher_client.to(self.device)
                    teacher_server = teacher_server.to(self.device)
                    
                    _, teacher_shared_features, _ = teacher_client(data)
                    teacher_logits = teacher_server(teacher_shared_features)
                    
                    teacher_logits_list.append(teacher_logits)
                    weights_list.append(teacher_weights[cluster_id])
            
            # 计算蒸馏损失
            total_batch_loss, loss_info = self.distillation_loss(
                student_logits, teacher_logits_list, weights_list, target
            )
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(student_client_model.parameters()) + list(student_server_model.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            
            # 累计损失
            total_loss += total_batch_loss.item()
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    loss_details_sum[key] += value
            batch_count += 1
        
        # 计算平均损失
        avg_loss = total_loss / max(1, batch_count)
        avg_loss_details = {key: value / max(1, batch_count) for key, value in loss_details_sum.items()}
        
        return avg_loss, avg_loss_details
    
    def _validate_student_model(self, student_client_model, student_server_model, validation_loader):
        """验证学生模型"""
        student_client_model.eval()
        student_server_model.eval()
        
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                _, shared_features, _ = student_client_model(data)
                logits = student_server_model(shared_features)
                
                # 计算损失和准确率
                loss = nn.CrossEntropyLoss()(logits, target)
                val_loss += loss.item()
                
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        # 恢复训练模式
        student_client_model.train()
        student_server_model.train()
        
        accuracy = 100.0 * correct / max(1, total)
        avg_val_loss = val_loss / max(1, len(validation_loader))
        
        return accuracy, avg_val_loss

class KnowledgeDistillationAggregator:
    """知识蒸馏聚合器"""
    
    def __init__(self, client_data_sizes, cluster_manager, dataset_name, data_dir, device='cuda'):
        """
        Args:
            client_data_sizes: 客户端数据大小字典
            cluster_manager: 聚类管理器
            dataset_name: 数据集名称
            data_dir: 数据目录
            device: 设备
        """
        self.device = device
        self.client_data_sizes = client_data_sizes
        self.cluster_manager = cluster_manager
        self.cluster_assignments = cluster_manager.get_cluster_assignments()
        self.cluster_info = cluster_manager.get_cluster_info()
        
        # 蒸馏数据管理器
        self.distillation_manager = DistillationDataManager(
            dataset_name=dataset_name,
            data_dir=data_dir,
            distillation_ratio=0.15,
            batch_size=128
        )
        
        # 学生模型训练器
        self.student_trainer = StudentModelTrainer(device=device)
        
        # 权重计算器
        self.weight_calculator = DynamicWeightCalculator(weight_strategy='consistency_based')
        
        self.logger = logging.getLogger("KnowledgeDistillationAggregator")
        
        # 创建蒸馏数据集
        self._initialize_distillation_data()
    
    def _initialize_distillation_data(self):
        """初始化蒸馏数据集"""
        try:
            self.distillation_loader, self.validation_loader = self.distillation_manager.create_distillation_dataset()
            self.logger.info("蒸馏数据集初始化成功")
        except Exception as e:
            self.logger.error(f"蒸馏数据集初始化失败: {str(e)}")
            raise
    
    def aggregate(self, shared_states, server_states):
        """
        执行知识蒸馏聚合
        
        Args:
            shared_states: 字典，{client_id: shared_layer_state_dict}
            server_states: 字典，{client_id: server_model_state_dict}
            
        Returns:
            global_shared_layers: 最终全局共享层参数
            global_server_model: 最终全局服务器模型参数
        """
        self.logger.info("开始知识蒸馏聚合过程...")
        
        # 第一级聚合：组内聚合（使用传统方法）
        cluster_shared_models, cluster_server_models = self._first_level_aggregation(
            shared_states, server_states
        )
        
        # 第二级聚合：多师生知识蒸馏
        global_shared_layers, global_server_model = self._second_level_distillation(
            cluster_shared_models, cluster_server_models
        )
        
        return global_shared_layers, global_server_model
    
    def _first_level_aggregation(self, shared_states, server_states):
        """第一级聚合：组内传统加权平均聚合"""
        self.logger.info("执行第一级聚合（组内聚合）...")
        
        cluster_shared_models = {}
        cluster_server_models = {}
        
        # 按组聚合
        for cluster_id, cluster_info in self.cluster_info.items():
            client_list = cluster_info['clients']
            self.logger.info(f"聚合组 {cluster_id}，客户端: {client_list}")
            
            # 计算组内权重（基于数据量）
            cluster_weights = {}
            total_cluster_data = sum(self.client_data_sizes[cid] for cid in client_list)
            
            for client_id in client_list:
                cluster_weights[client_id] = self.client_data_sizes[client_id] / total_cluster_data
            
            # 聚合组内共享层
            cluster_shared_states = {cid: shared_states[cid] for cid in client_list if cid in shared_states}
            cluster_shared_models[cluster_id] = self._aggregate_parameters(
                cluster_shared_states, cluster_weights
            )
            
            # 聚合组内服务器模型
            cluster_server_states = {cid: server_states[cid] for cid in client_list if cid in server_states}
            cluster_server_models[cluster_id] = self._aggregate_parameters(
                cluster_server_states, cluster_weights
            )
        
        return cluster_shared_models, cluster_server_models
    
    def _second_level_distillation(self, cluster_shared_models, cluster_server_models):
        """第二级聚合：知识蒸馏"""
        self.logger.info("执行第二级聚合（知识蒸馏）...")
        
        # 计算教师权重
        teacher_weights = self.weight_calculator.calculate_teacher_weights(self.cluster_info)
        self.logger.info(f"教师模型权重: {teacher_weights}")
        
        # 创建教师模型
        teacher_models = self._create_teacher_models(cluster_shared_models, cluster_server_models)
        
        # 创建学生模型
        student_client_model, student_server_model = self._create_student_model(
            cluster_shared_models, cluster_server_models
        )
        
        # 执行知识蒸馏训练
        training_stats = self.student_trainer.train_student_model(
            student_client_model=student_client_model,
            student_server_model=student_server_model,
            teacher_models=teacher_models,
            teacher_weights=teacher_weights,
            distillation_loader=self.distillation_loader,
            validation_loader=self.validation_loader,
            num_epochs=8,
            early_stop_patience=3
        )
        
        # 提取最终的全局模型参数
        global_shared_layers = {}
        for name, param in student_client_model.named_parameters():
            if 'shared_base' in name:
                global_shared_layers[name] = param.data.clone().cpu()
        
        global_server_model = {}
        for name, param in student_server_model.named_parameters():
            global_server_model[name] = param.data.clone().cpu()
        
        self.logger.info(f"知识蒸馏完成，最佳验证准确率: {training_stats['best_val_accuracy']:.2f}%")
        
        return global_shared_layers, global_server_model
    
    def _create_teacher_models(self, cluster_shared_models, cluster_server_models):
        """创建教师模型实例"""
        from model.resnet import TierAwareClientModel, EnhancedServerModel
        
        teacher_models = {}
        
        for cluster_id in cluster_shared_models.keys():
            # 创建教师客户端模型
            teacher_client = TierAwareClientModel(
                num_classes=10,  # 假设10个类别
                tier=2,  # 使用中等配置
                model_type='resnet56',
                input_channels=3 if self.distillation_manager.dataset_name == 'cifar10' else 1
            ).to(self.device)
            
            # 创建教师服务器模型
            teacher_server = EnhancedServerModel(
                num_classes=10,
                model_type='resnet56',
                input_channels=3 if self.distillation_manager.dataset_name == 'cifar10' else 1
            ).to(self.device)
            
            # 加载聚合后的参数（使用strict=False避免缺少keys的问题）
            missing_keys, unexpected_keys = teacher_client.load_state_dict(cluster_shared_models[cluster_id], strict=False)
            if missing_keys:
                logging.debug(f"教师客户端模型缺少keys: {missing_keys}")
            if unexpected_keys:
                logging.debug(f"教师客户端模型多余keys: {unexpected_keys}")
                
            missing_keys, unexpected_keys = teacher_server.load_state_dict(cluster_server_models[cluster_id], strict=False)
            if missing_keys:
                logging.debug(f"教师服务器模型缺少keys: {missing_keys}")
            if unexpected_keys:
                logging.debug(f"教师服务器模型多余keys: {unexpected_keys}")
            
            teacher_models[cluster_id] = (teacher_client, teacher_server)
        
        return teacher_models
    
    def _create_student_model(self, cluster_shared_models, cluster_server_models):
        """创建学生模型，初始化为教师模型的加权平均"""
        from model.resnet import TierAwareClientModel, EnhancedServerModel
        
        # 创建学生模型
        student_client = TierAwareClientModel(
            num_classes=10,
            tier=2,
            model_type='resnet56',
            input_channels=3 if self.distillation_manager.dataset_name == 'cifar10' else 1
        ).to(self.device)
        
        student_server = EnhancedServerModel(
            num_classes=10,
            model_type='resnet56',
            input_channels=3 if self.distillation_manager.dataset_name == 'cifar10' else 1
        ).to(self.device)
        
        # 使用教师模型的加权平均初始化学生模型
        teacher_weights = self.weight_calculator.calculate_teacher_weights(self.cluster_info)
        
        # 初始化共享层
        student_shared_state = self._aggregate_parameters(cluster_shared_models, teacher_weights)
        missing_keys, unexpected_keys = student_client.load_state_dict(student_shared_state, strict=False)
        if missing_keys:
            logging.debug(f"学生客户端模型缺少keys: {missing_keys}")
        
        # 初始化服务器模型
        student_server_state = self._aggregate_parameters(cluster_server_models, teacher_weights)
        missing_keys, unexpected_keys = student_server.load_state_dict(student_server_state, strict=False)
        if missing_keys:
            logging.debug(f"学生服务器模型缺少keys: {missing_keys}")
        
        return student_client, student_server
    
    def _aggregate_parameters(self, state_dicts, weights):
        """基于权重聚合参数和缓冲区"""
        if not state_dicts:
            return {}
        
        # 获取第一个状态字典的键
        first_id = next(iter(state_dicts.keys()))
        param_keys = state_dicts[first_id].keys()
        
        # 初始化聚合结果
        aggregated_state = {}
        
        # 对每个参数和缓冲区进行加权平均
        for key in param_keys:
            weighted_sum = None
            total_weight = 0.0
            
            # 加权累加
            for model_id, state_dict in state_dicts.items():
                if key in state_dict and model_id in weights:
                    weight = weights[model_id]
                    total_weight += weight
                    
                    # 确保参数在正确的设备上
                    param_tensor = state_dict[key].to(self.device)
                    
                    # 对于BatchNorm的num_batches_tracked，使用最大值而不是平均值
                    if 'num_batches_tracked' in key:
                        if weighted_sum is None:
                            weighted_sum = param_tensor
                        else:
                            weighted_sum = torch.max(weighted_sum, param_tensor)
                    else:
                        # 正常的参数和统计量使用加权平均
                        if weighted_sum is None:
                            weighted_sum = weight * param_tensor
                        else:
                            weighted_sum += weight * param_tensor
            
            # 计算最终结果
            if weighted_sum is not None and total_weight > 0:
                if 'num_batches_tracked' in key:
                    aggregated_state[key] = weighted_sum
                else:
                    aggregated_state[key] = weighted_sum / total_weight
            else:
                # 如果无法聚合，使用第一个模型的参数
                aggregated_state[key] = state_dicts[first_id][key].to(self.device)
        
        return aggregated_state
    
    def update_distillation_config(self, temperature=None, alpha=None, beta=None):
        """更新蒸馏配置"""
        if temperature is not None:
            self.student_trainer.distillation_loss.update_temperature(temperature)
        if alpha is not None and beta is not None:
            self.student_trainer.distillation_loss.update_loss_weights(alpha, beta)
        
        self.logger.info("蒸馏配置已更新")