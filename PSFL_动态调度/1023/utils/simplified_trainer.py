import torch
import torch.nn as nn
import time
import logging
from utils.tierhfl_loss import EnhancedStagedLoss
from model.resnet import get_resnet_num_blocks, get_tier_shared_block_counts

class SimplifiedSerialTrainer:
    """简化版串行训练器，适配新的模型架构"""
    
    def __init__(self, client_manager, client_models, server_models, device="cuda"):
        self.client_manager = client_manager
        self.client_models = client_models
        self.server_models = server_models
        self.device = device
        
        # 增强损失函数
        self.enhanced_loss = EnhancedStagedLoss()

    def _simulate_resource_delay(self, client, measured_train_time: float, model_type: str = "resnet56") -> float:
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

        tier = int(getattr(client, "tier", 4) or 4)
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
        
    def execute_round(self, round_idx, total_rounds, selected_client_ids=None, training_phase=None):
        """执行一轮训练"""
        start_time = time.time()
        
        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        server_states = {}
        
        if selected_client_ids is None:
            client_ids = sorted(self.client_models.keys())
        else:
            client_ids = sorted(list(selected_client_ids))

        phase = str(training_phase or "initial")
        for client_id in client_ids:
            client = self.client_manager.get_client(client_id)
            if not client:
                continue
            
            client_model = self.client_models[client_id].to(self.device)
            server_model = self.server_models[client_id].to(self.device)
            client.model = client_model
            
            if phase == "initial":
                train_result = self._train_global_only(client, client_model, server_model, round_idx, total_rounds)
            else:
                train_result = self._train_joint(client, client_model, server_model, round_idx, total_rounds, phase)

            measured_time_cost = float(train_result.get("time_cost", 0.0) or 0.0)
            simulated_delay = self._simulate_resource_delay(client, measured_time_cost, model_type=getattr(client_model, "model_type", "resnet56"))
            simulated_time_cost = measured_time_cost + simulated_delay
            train_result["raw_time_cost"] = measured_time_cost
            train_result["simulated_delay"] = simulated_delay
            train_result["simulated_time_cost"] = simulated_time_cost
            train_result["time_cost"] = measured_time_cost

            train_acc = train_result.get("global_accuracy", None)
            local_train_acc = train_result.get("local_accuracy", None)
            if local_train_acc is not None and train_acc is not None:
                logging.info(
                    "客户端 %s | 设备tier=%s | 拆分点=%s | 阶段=%s | 本地训练准确率=%.2f%% | 全局训练准确率=%.2f%% | 训练时间=%.3fs",
                    client_id,
                    getattr(client, "device_tier", None),
                    getattr(client, "tier", None),
                    phase,
                    float(local_train_acc),
                    float(train_acc),
                    measured_time_cost,
                )
            elif train_acc is not None:
                logging.info(
                    "客户端 %s | 设备tier=%s | 拆分点=%s | 阶段=%s | 全局训练准确率=%.2f%% | 训练时间=%.3fs",
                    client_id,
                    getattr(client, "device_tier", None),
                    getattr(client, "tier", None),
                    phase,
                    float(train_acc),
                    measured_time_cost,
                )
            else:
                logging.info(
                    "客户端 %s | 设备tier=%s | 拆分点=%s | 阶段=%s | 训练时间=%.3fs",
                    client_id,
                    getattr(client, "device_tier", None),
                    getattr(client, "tier", None),
                    phase,
                    measured_time_cost,
                )
            
            # 评估客户端
            eval_result = self._evaluate_client(client, client_model, server_model)
            
            # 保存结果
            train_results[client_id] = train_result
            eval_results[client_id] = eval_result
            
            # 保存共享层状态（包括参数和缓冲区）
            shared_state = {}
            for name, param in client_model.named_parameters():
                if 'shared_base' in name:
                    shared_state[name] = param.data.clone().cpu()
            # 添加缓冲区
            for name, buffer in client_model.named_buffers():
                if 'shared_base' in name:
                    shared_state[name] = buffer.clone().cpu()
            shared_states[client_id] = shared_state
            
            # 保存服务器模型状态（包括参数和缓冲区）
            server_state = {}
            for name, param in server_model.named_parameters():
                server_state[name] = param.data.clone().cpu()
            for name, buffer in server_model.named_buffers():
                server_state[name] = buffer.clone().cpu()
            server_states[client_id] = server_state
            
            # 更新客户端模型
            self.client_models[client_id] = client_model.cpu()
            self.server_models[client_id] = server_model.cpu()
            
            torch.cuda.empty_cache()
        
        training_time = time.time() - start_time
        return train_results, eval_results, shared_states, server_states, training_time
    
    def _train_global_only(self, client, client_model, server_model, round_idx, total_rounds):
        """只训练全局路径：客户端共享层 + 服务器模型"""
        start_time = time.time()
        
        for name, param in client_model.named_parameters():
            param.requires_grad = ('shared_base' in name)
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n and p.requires_grad], 
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
            
            shared_features = client_model.shared_base(data)
            global_logits = server_model(shared_features)
            
            # 使用增强损失函数
            total_loss, global_loss, feature_importance_loss = self.enhanced_loss.stage1_loss(
                global_logits, target, shared_features
            )
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in client_model.named_parameters() if 'shared_base' in n and p.requires_grad], 
                max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=1.0)
            
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

    def _train_joint(self, client, client_model, server_model, round_idx, total_rounds, phase: str):
        start_time = time.time()

        for _, param in client_model.named_parameters():
            param.requires_grad = True
        for _, param in server_model.named_parameters():
            param.requires_grad = True

        client_model.train()
        server_model.train()

        if str(phase) == "fine_tuning":
            alpha = 0.7
        else:
            alpha = float(getattr(client, "alpha", 0.5) or 0.5)

        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if "shared_base" in n and p.requires_grad],
            lr=client.lr * 0.5,
        )
        local_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if "shared_base" not in n and p.requires_grad],
            lr=client.lr,
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)

        stats = {
            "local_loss": 0.0,
            "global_loss": 0.0,
            "balance_loss": 0.0,
            "total_loss": 0.0,
            "local_correct": 0,
            "global_correct": 0,
            "total": 0,
            "batch_count": 0,
        }

        for _ in range(int(getattr(client, "local_epochs", 1) or 1)):
            for _, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)

                shared_optimizer.zero_grad()
                local_optimizer.zero_grad()
                server_optimizer.zero_grad()

                local_logits, shared_features, _ = client_model(data)
                global_logits = server_model(shared_features)

                total_loss, local_loss, global_loss, balance_loss = self.enhanced_loss.stage2_3_loss(
                    local_logits, global_logits, target, shared_features=shared_features, alpha=float(alpha)
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if "shared_base" in n and p.requires_grad],
                    max_norm=1.0,
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if "shared_base" not in n and p.requires_grad],
                    max_norm=1.0,
                )
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=1.0)

                shared_optimizer.step()
                local_optimizer.step()
                server_optimizer.step()

                stats["local_loss"] += float(local_loss.item())
                stats["global_loss"] += float(global_loss.item())
                stats["balance_loss"] += float(balance_loss.item())
                stats["total_loss"] += float(total_loss.item())
                stats["batch_count"] += 1

                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                stats["local_correct"] += int(local_pred.eq(target).sum().item())
                stats["global_correct"] += int(global_pred.eq(target).sum().item())
                stats["total"] += int(target.size(0))

        if stats["batch_count"] > 0:
            stats["local_loss"] /= stats["batch_count"]
            stats["global_loss"] /= stats["batch_count"]
            stats["balance_loss"] /= stats["batch_count"]
            stats["total_loss"] /= stats["batch_count"]

        if stats["total"] > 0:
            local_accuracy = 100.0 * stats["local_correct"] / stats["total"]
            global_accuracy = 100.0 * stats["global_correct"] / stats["total"]
        else:
            local_accuracy = 0.0
            global_accuracy = 0.0

        return {
            "local_loss": stats["local_loss"],
            "global_loss": stats["global_loss"],
            "balance_loss": stats["balance_loss"],
            "total_loss": stats["total_loss"],
            "local_accuracy": local_accuracy,
            "global_accuracy": global_accuracy,
            "alpha": float(alpha),
            "time_cost": time.time() - start_time,
        }
    
    def _evaluate_client(self, client, client_model, server_model):
        """评估客户端模型"""
        # 设置评估模式
        client_model.eval()
        server_model.eval()
        
        local_train_correct = 0
        local_test_correct = 0
        split_train_correct = 0
        split_test_correct = 0
        train_total = 0
        test_total = 0
        
        with torch.no_grad():
            # 评估训练集
            for data, target in client.train_data:
                data, target = data.to(self.device), target.to(self.device)
                
                local_logits, shared_features, _ = client_model(data)
                global_logits = server_model(shared_features)
                
                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                local_train_correct += local_pred.eq(target).sum().item()
                split_train_correct += global_pred.eq(target).sum().item()
                train_total += target.size(0)
            
            # 评估测试集
            for data, target in client.test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                local_logits, shared_features, _ = client_model(data)
                global_logits = server_model(shared_features)
                
                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                local_test_correct += local_pred.eq(target).sum().item()
                split_test_correct += global_pred.eq(target).sum().item()
                test_total += target.size(0)
        
        # 计算准确率
        local_train_accuracy = 100.0 * local_train_correct / max(1, train_total)
        local_test_accuracy = 100.0 * local_test_correct / max(1, test_total)
        split_train_accuracy = 100.0 * split_train_correct / max(1, train_total)
        split_test_accuracy = 100.0 * split_test_correct / max(1, test_total)
        
        return {
            "local_train_accuracy": local_train_accuracy,
            "local_test_accuracy": local_test_accuracy,
            "split_train_accuracy": split_train_accuracy,
            "split_test_accuracy": split_test_accuracy,
            "train_samples": train_total,
            "test_samples": test_total,
        }
    
    def update_global_models(self, global_shared_layers, global_server_model):
        """更新所有客户端的共享层和服务器模型（包括参数和缓冲区）"""
        
        logging.info("开始更新全局模型...")
        logging.info(f"全局共享层键数: {len(global_shared_layers)}")
        logging.info(f"全局服务器模型键数: {len(global_server_model)}")
        
        # 更新客户端共享层
        for client_id, model in self.client_models.items():
            # 更新参数
            param_updated = 0
            for name, param in model.named_parameters():
                if 'shared_base' in name and name in global_shared_layers:
                    param.data.copy_(global_shared_layers[name])
                    param_updated += 1
            
            # 更新缓冲区
            buffer_updated = 0
            for name, buffer in model.named_buffers():
                if 'shared_base' in name and name in global_shared_layers:
                    buffer.copy_(global_shared_layers[name])
                    buffer_updated += 1
            
            logging.debug(f"客户端 {client_id} 共享层: 更新了 {param_updated} 个参数, {buffer_updated} 个缓冲区")
        
        # 更新服务器模型
        for client_id, model in self.server_models.items():
            # 更新参数
            param_updated = 0
            for name, param in model.named_parameters():
                if name in global_server_model:
                    param.data.copy_(global_server_model[name])
                    param_updated += 1
            
            # 更新缓冲区
            buffer_updated = 0
            for name, buffer in model.named_buffers():
                if name in global_server_model:
                    buffer.copy_(global_server_model[name])
                    buffer_updated += 1
            
            logging.debug(f"客户端 {client_id} 服务器模型: 更新了 {param_updated} 个参数, {buffer_updated} 个缓冲区")
        
        logging.info("已更新所有客户端的共享层和服务器模型（包括参数和缓冲区）")
