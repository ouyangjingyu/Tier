import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import logging
from utils.tierhfl_loss import EnhancedStagedLoss

class SimplifiedSerialTrainer:
    """简化版串行训练器，每个客户端对应一个服务器模型"""
    
    def __init__(self, client_manager, client_models, server_models, device="cuda"):
        self.client_manager = client_manager
        self.client_models = client_models
        self.server_models = server_models
        self.device = device
        
        # 增强损失函数
        self.enhanced_loss = EnhancedStagedLoss()
        
    def execute_round(self, round_idx, total_rounds, training_phase="alternating"):
        """执行一轮训练"""
        start_time = time.time()
        
        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        server_states = {}
        
        # 串行训练每个客户端
        for client_id in range(len(self.client_models)):
            logging.info(f"训练客户端 {client_id}")
            
            client = self.client_manager.get_client(client_id)
            if not client:
                continue
            
            client_model = self.client_models[client_id].to(self.device)
            server_model = self.server_models[client_id].to(self.device)
            client.model = client_model
            
            # 根据训练阶段执行训练
            if training_phase == "initial":
                train_result = self._train_initial_phase(
                    client, client_model, server_model, round_idx, total_rounds)
            elif training_phase == "alternating":
                train_result = self._train_alternating_phase(
                    client, client_model, server_model, round_idx, total_rounds)
            else:  # fine_tuning
                train_result = self._train_fine_tuning_phase(
                    client, client_model, server_model, round_idx, total_rounds)
            
            # 评估客户端
            eval_result = self._evaluate_client(client, client_model, server_model)
            
            # 保存结果
            train_results[client_id] = train_result
            eval_results[client_id] = eval_result
            
            # 保存共享层状态
            shared_state = {}
            for name, param in client_model.named_parameters():
                if 'shared_base' in name:
                    shared_state[name] = param.data.clone().cpu()
            shared_states[client_id] = shared_state
            
            # 保存服务器模型状态
            server_states[client_id] = {
                name: param.data.clone().cpu() 
                for name, param in server_model.named_parameters()
            }
            
            # 更新客户端模型
            self.client_models[client_id] = client_model.cpu()
            self.server_models[client_id] = server_model.cpu()
            
            torch.cuda.empty_cache()
        
        training_time = time.time() - start_time
        return train_results, eval_results, shared_states, server_states, training_time
    
    def _train_initial_phase(self, client, client_model, server_model, round_idx, total_rounds):
        """阶段1: 初始特征学习阶段"""
        start_time = time.time()
        
        # 冻结个性化路径
        for name, param in client_model.named_parameters():
            if 'personalized_path' in name or 'local_classifier' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=client.lr * 0.5
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'feature_importance_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(client.train_data):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除梯度
            shared_optimizer.zero_grad()
            server_optimizer.zero_grad()
            
            # 前向传播
            local_logits, shared_features, personal_features = client_model(data)
            global_logits = server_model(shared_features)
            
            # 使用增强损失函数
            total_loss, global_loss, feature_importance_loss = self.enhanced_loss.stage1_loss(
                global_logits, target, shared_features
            )
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                max_norm=1.0
            )
            
            # 更新参数
            shared_optimizer.step()
            server_optimizer.step()
            
            # 更新统计信息
            stats['global_loss'] += global_loss.item()
            stats['feature_importance_loss'] += feature_importance_loss.item()
            stats['total_loss'] += total_loss.item()
            stats['batch_count'] += 1
            
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(target).sum().item()
            stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'feature_importance_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        # 解冻所有层
        for name, param in client_model.named_parameters():
            param.requires_grad = True
        
        return {
            'global_loss': stats['global_loss'],
            'feature_importance_loss': stats['feature_importance_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }
    
    def _train_alternating_phase(self, client, client_model, server_model, round_idx, total_rounds):
        """阶段2: 交替训练阶段"""
        # 第一部分: 训练个性化路径 (1 epoch)
        personal_result = self._train_personal_path(
            client, client_model, client.lr * 0.05, round_idx, total_rounds)
        
        # 第二部分: 训练全局路径 (2 epochs)
        global_result = self._train_global_path(
            client, client_model, server_model, 
            client.lr * 0.1, round_idx, total_rounds, epochs=2)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'local_loss': personal_result['local_loss'],
            'balance_loss': global_result.get('balance_loss', 0.0),
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result
    
    def _train_fine_tuning_phase(self, client, client_model, server_model, round_idx, total_rounds):
        """阶段3: 精细调整阶段"""
        # 全局路径训练
        global_result = self._train_global_path(
            client, client_model, server_model, 
            client.lr * 0.02, round_idx, total_rounds, epochs=2)
        
        # 个性化路径训练
        personal_result = self._train_personal_path(
            client, client_model, client.lr * 0.01, 
            round_idx, total_rounds)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'local_loss': personal_result['local_loss'],
            'balance_loss': global_result.get('balance_loss', 0.0),
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result
    
    def _train_global_path(self, client, client_model, server_model, shared_lr, round_idx, total_rounds, epochs=1):
        """全局路径训练"""
        start_time = time.time()
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=shared_lr
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'local_loss': 0.0,
            'balance_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 计算alpha值
        progress = round_idx / total_rounds
        alpha = 0.3 + 0.4 * progress
        
        # 训练循环
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                global_logits = server_model(shared_features)
                
                # 计算损失
                local_loss = F.cross_entropy(local_logits, target)
                global_loss = F.cross_entropy(global_logits, target)
                
                # 简化的特征平衡损失
                balance_loss = torch.tensor(0.0, device=global_logits.device)
                
                # 使用增强损失函数
                total_loss, local_loss_calc, global_loss_calc, balance_loss = self.enhanced_loss.stage2_3_loss(
                    local_logits, global_logits, target, 
                    personal_gradients=None, global_gradients=None,
                    shared_features=shared_features, alpha=alpha
                )
                
                # 清除梯度
                shared_optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                    max_norm=0.5
                )
                
                # 更新参数
                shared_optimizer.step()
                server_optimizer.step()
                
                # 更新统计信息
                stats['global_loss'] += global_loss.item()
                stats['local_loss'] += local_loss.item()
                stats['balance_loss'] += balance_loss.item()
                stats['total_loss'] += total_loss.item()
                stats['batch_count'] += 1
                
                _, pred = global_logits.max(1)
                stats['correct'] += pred.eq(target).sum().item()
                stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'local_loss', 'balance_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        return {
            'global_loss': stats['global_loss'],
            'local_loss': stats['local_loss'],
            'balance_loss': stats['balance_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }
    
    def _train_personal_path(self, client, client_model, shared_lr, round_idx, total_rounds):
        """个性化路径训练"""
        start_time = time.time()
        
        client_model.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n],
            lr=shared_lr
        )
        personal_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() 
            if 'shared_base' not in n and p.requires_grad],
            lr=client.lr
        )
        
        # 统计信息
        stats = {
            'local_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for epoch in range(client.local_epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                shared_optimizer.zero_grad()
                personal_optimizer.zero_grad()
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                local_loss = F.cross_entropy(local_logits, target)
                
                # 反向传播
                local_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                    max_norm=0.3
                )
                
                # 更新参数
                shared_optimizer.step()
                personal_optimizer.step()
                
                # 更新统计信息
                stats['local_loss'] += local_loss.item()
                stats['batch_count'] += 1
                
                _, pred = local_logits.max(1)
                stats['correct'] += pred.eq(target).sum().item()
                stats['total'] += target.size(0)
        
        # 计算结果
        avg_local_loss = stats['local_loss'] / max(1, stats['batch_count'])
        local_accuracy = 100.0 * stats['correct'] / max(1, stats['total'])
        
        return {
            'local_loss': avg_local_loss,
            'local_accuracy': local_accuracy,
            'time_cost': time.time() - start_time
        }
    
    def _evaluate_client(self, client, client_model, server_model):
        """评估客户端模型"""
        # 设置评估模式
        client_model.eval()
        server_model.eval()
        
        # 本地模型统计
        local_train_correct = 0
        local_test_correct = 0
        local_train_total = 0
        local_test_total = 0
        
        # 拆分模型统计
        split_train_correct = 0
        split_test_correct = 0
        split_train_total = 0
        split_test_total = 0
        
        with torch.no_grad():
            # 评估训练集
            for data, target in client.train_data:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, _ = client_model(data)
                global_logits = server_model(shared_features)
                
                # 本地分类器准确率
                _, local_pred = local_logits.max(1)
                local_train_correct += local_pred.eq(target).sum().item()
                local_train_total += target.size(0)
                
                # 拆分学习准确率
                _, global_pred = global_logits.max(1)
                split_train_correct += global_pred.eq(target).sum().item()
                split_train_total += target.size(0)
            
            # 评估测试集
            for data, target in client.test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, _ = client_model(data)
                global_logits = server_model(shared_features)
                
                # 本地分类器准确率
                _, local_pred = local_logits.max(1)
                local_test_correct += local_pred.eq(target).sum().item()
                local_test_total += target.size(0)
                
                # 拆分学习准确率
                _, global_pred = global_logits.max(1)
                split_test_correct += global_pred.eq(target).sum().item()
                split_test_total += target.size(0)
        
        # 计算准确率
        local_train_accuracy = 100.0 * local_train_correct / max(1, local_train_total)
        local_test_accuracy = 100.0 * local_test_correct / max(1, local_test_total)
        split_train_accuracy = 100.0 * split_train_correct / max(1, split_train_total)
        split_test_accuracy = 100.0 * split_test_correct / max(1, split_test_total)
        
        return {
            'local_train_accuracy': local_train_accuracy,
            'local_test_accuracy': local_test_accuracy,
            'split_train_accuracy': split_train_accuracy,
            'split_test_accuracy': split_test_accuracy,
            'train_samples': local_train_total,
            'test_samples': local_test_total
        }
    
    def update_global_models(self, global_shared_layers, global_server_model):
        """更新所有客户端的共享层和服务器模型"""
        # 更新客户端共享层
        for client_id, model in self.client_models.items():
            for name, param in model.named_parameters():
                if 'shared_base' in name and name in global_shared_layers:
                    param.data.copy_(global_shared_layers[name])
        
        # 更新服务器模型
        for client_id, model in self.server_models.items():
            for name, param in model.named_parameters():
                if name in global_server_model:
                    param.data.copy_(global_server_model[name])
        
        logging.info("已更新所有客户端的共享层和服务器模型")