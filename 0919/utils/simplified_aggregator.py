import torch
import copy
import logging
import numpy as np
from collections import defaultdict

class HierarchicalAggregator:
    """分层聚合器，实现二级聚合策略"""
    
    def __init__(self, client_data_sizes, cluster_manager, device='cuda'):
        """
        Args:
            client_data_sizes: 字典，{client_id: data_size}
            cluster_manager: 客户端聚类管理器
            device: 设备
        """
        self.device = device
        self.client_data_sizes = client_data_sizes
        self.cluster_manager = cluster_manager
        self.cluster_assignments = cluster_manager.get_cluster_assignments()
        self.cluster_info = cluster_manager.get_cluster_info()
        
        # 计算客户端聚合权重
        self.client_weights = {}
        total_data_size = sum(client_data_sizes.values())
        for client_id, data_size in client_data_sizes.items():
            self.client_weights[client_id] = data_size / total_data_size
        
        logging.info(f"初始化分层聚合器，客户端权重: {self.client_weights}")
        logging.info(f"聚类分配: {self.cluster_assignments}")
    
    def aggregate(self, shared_states, server_states):
        """
        执行二级聚合
        
        Args:
            shared_states: 字典，{client_id: shared_layer_state_dict}
            server_states: 字典，{client_id: server_model_state_dict}
            
        Returns:
            global_shared_layers: 最终全局共享层参数
            global_server_model: 最终全局服务器模型参数
        """
        logging.info("开始二级聚合过程...")
        
        # 第一级聚合：组内聚合
        cluster_shared_models, cluster_server_models = self._first_level_aggregation(
            shared_states, server_states
        )
        
        # 第二级聚合：组间聚合
        global_shared_layers, global_server_model = self._second_level_aggregation(
            cluster_shared_models, cluster_server_models
        )
        
        return global_shared_layers, global_server_model
    
    def _first_level_aggregation(self, shared_states, server_states):
        """
        第一级聚合：组内聚合
        在每个组内使用基于数据量的加权平均
        """
        logging.info("执行第一级聚合（组内聚合）...")
        
        cluster_shared_models = {}
        cluster_server_models = {}
        
        # 按组聚合
        for cluster_id, cluster_info in self.cluster_info.items():
            client_list = cluster_info['clients']
            logging.info(f"聚合组 {cluster_id}，客户端: {client_list}")
            
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
            
            logging.info(f"组 {cluster_id} 聚合完成，使用权重: {cluster_weights}")
        
        return cluster_shared_models, cluster_server_models
    
    def _second_level_aggregation(self, cluster_shared_models, cluster_server_models):
        """
        第二级聚合：组间聚合
        使用基于覆盖度和一致性的智能权重
        """
        logging.info("执行第二级聚合（组间聚合）...")
        
        # 计算组间聚合权重
        cluster_weights = self._calculate_cluster_weights(cluster_shared_models, cluster_server_models)
        
        # 聚合共享层
        global_shared_layers = self._aggregate_parameters(
            cluster_shared_models, cluster_weights
        )
        
        # 聚合服务器模型
        global_server_model = self._aggregate_parameters(
            cluster_server_models, cluster_weights
        )
        
        logging.info(f"第二级聚合完成，使用组权重: {cluster_weights}")
        
        return global_shared_layers, global_server_model
    
    def _calculate_cluster_weights(self, cluster_shared_models, cluster_server_models, alpha=0.6):
        """
        计算组间聚合权重
        结合覆盖度权重和一致性权重
        
        Args:
            cluster_shared_models: 组内聚合的共享层模型
            cluster_server_models: 组内聚合的服务器模型  
            alpha: 覆盖度权重和一致性权重的平衡因子
        """
        # 1. 计算覆盖度权重
        coverage_weights = self.cluster_manager.calculate_cluster_coverage_weights()
        
        # 2. 计算一致性权重
        consistency_weights = self._calculate_consistency_weights(cluster_shared_models, cluster_server_models)
        
        # 3. 组合权重
        cluster_weights = {}
        cluster_ids = list(coverage_weights.keys())
        
        for cluster_id in cluster_ids:
            coverage_w = coverage_weights.get(cluster_id, 0.0)
            consistency_w = consistency_weights.get(cluster_id, 0.0)
            cluster_weights[cluster_id] = alpha * coverage_w + (1 - alpha) * consistency_w
        
        # 4. 归一化权重
        total_weight = sum(cluster_weights.values())
        if total_weight > 0:
            for cluster_id in cluster_weights:
                cluster_weights[cluster_id] /= total_weight
        else:
            # 如果权重都为0，使用等权重
            num_clusters = len(cluster_ids)
            for cluster_id in cluster_ids:
                cluster_weights[cluster_id] = 1.0 / num_clusters
        
        logging.info(f"覆盖度权重: {coverage_weights}")
        logging.info(f"一致性权重: {consistency_weights}")
        logging.info(f"最终组权重 (α={alpha}): {cluster_weights}")
        
        return cluster_weights
    
    def _calculate_consistency_weights(self, cluster_shared_models, cluster_server_models):
        """
        计算组内一致性权重
        基于组内客户端模型参数的方差
        """
        consistency_weights = {}
        
        for cluster_id, cluster_info in self.cluster_info.items():
            client_list = cluster_info['clients']
            
            if len(client_list) <= 1:
                # 单客户端组，一致性最高
                consistency_weights[cluster_id] = 1.0
                continue
            
            # 计算组内客户端模型的参数方差
            variances = []
            
            # 检查共享层参数方差
            if cluster_id in cluster_shared_models:
                shared_params = cluster_shared_models[cluster_id]
                for param_name, param_tensor in shared_params.items():
                    if param_tensor.numel() > 0:
                        param_var = torch.var(param_tensor.float()).item()
                        variances.append(param_var)
            
            # 计算平均方差
            if variances:
                avg_variance = np.mean(variances)
                # 一致性权重 = 1 / (1 + 方差)，方差越小权重越大
                consistency_weights[cluster_id] = 1.0 / (1.0 + avg_variance)
            else:
                consistency_weights[cluster_id] = 0.5  # 默认中等权重
        
        # 归一化一致性权重
        total_consistency = sum(consistency_weights.values())
        if total_consistency > 0:
            for cluster_id in consistency_weights:
                consistency_weights[cluster_id] /= total_consistency
        
        return consistency_weights
    
    def _aggregate_parameters(self, state_dicts, weights):
        """
        基于权重聚合参数和缓冲区
        
        Args:
            state_dicts: 字典，{id: state_dict}
            weights: 字典，{id: weight}
            
        Returns:
            aggregated_state_dict: 聚合后的参数字典
        """
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
    
    def update_client_data_sizes(self, new_data_sizes):
        """更新客户端数据量（如果动态变化）"""
        self.client_data_sizes = new_data_sizes
        
        # 重新计算客户端权重
        total_data_size = sum(new_data_sizes.values())
        self.client_weights = {}
        for client_id, data_size in new_data_sizes.items():
            self.client_weights[client_id] = data_size / total_data_size
        
        logging.info(f"更新客户端权重: {self.client_weights}")

# 为了向后兼容，保留原有的SimplifiedGlobalAggregator
class SimplifiedGlobalAggregator:
    """简化的全局聚合器，基于客户端数据量进行聚合（保留用于对比实验）"""
    
    def __init__(self, client_data_sizes, device='cuda'):
        """
        Args:
            client_data_sizes: 字典，{client_id: data_size}
            device: 设备
        """
        self.device = device
        self.client_data_sizes = client_data_sizes
        self.total_data_size = sum(client_data_sizes.values())
        
        # 计算聚合权重
        self.weights = {}
        for client_id, data_size in client_data_sizes.items():
            self.weights[client_id] = data_size / self.total_data_size
        
        logging.info(f"客户端聚合权重: {self.weights}")
    
    def aggregate(self, shared_states, server_states):
        """
        聚合共享层和服务器模型
        
        Args:
            shared_states: 字典，{client_id: shared_layer_state_dict}
            server_states: 字典，{client_id: server_model_state_dict}
            
        Returns:
            global_shared_layers: 聚合后的共享层参数
            global_server_model: 聚合后的服务器模型参数
        """
        # 聚合共享层
        global_shared_layers = self._aggregate_parameters(shared_states)
        
        # 聚合服务器模型
        global_server_model = self._aggregate_parameters(server_states)
        
        return global_shared_layers, global_server_model
    
    def _aggregate_parameters(self, state_dicts):
        """
        基于数据量权重聚合参数和缓冲区
        
        Args:
            state_dicts: 字典，{client_id: state_dict}
            
        Returns:
            aggregated_state_dict: 聚合后的参数字典
        """
        if not state_dicts:
            return {}
        
        # 获取第一个状态字典的键
        first_client_id = next(iter(state_dicts.keys()))
        param_keys = state_dicts[first_client_id].keys()
        
        # 初始化聚合结果
        aggregated_state = {}
        
        # 对每个参数和缓冲区进行加权平均
        for key in param_keys:
            weighted_sum = None
            total_weight = 0.0
            
            # 加权累加
            for client_id, state_dict in state_dicts.items():
                if key in state_dict and client_id in self.weights:
                    weight = self.weights[client_id]
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
                # 如果无法聚合，使用第一个客户端的参数
                aggregated_state[key] = state_dicts[first_client_id][key].to(self.device)
        
        return aggregated_state
    
    def update_client_data_sizes(self, new_data_sizes):
        """更新客户端数据量（如果动态变化）"""
        self.client_data_sizes = new_data_sizes
        self.total_data_size = sum(new_data_sizes.values())
        
        # 重新计算权重
        self.weights = {}
        for client_id, data_size in new_data_sizes.items():
            self.weights[client_id] = data_size / self.total_data_size
        
        logging.info(f"更新客户端聚合权重: {self.weights}")

# 知识蒸馏聚合器的接口类，用于统一接口
class KnowledgeDistillationAggregatorWrapper:
    """知识蒸馏聚合器包装类，提供统一接口"""
    
    def __init__(self, client_data_sizes, cluster_manager, dataset_name, data_dir, device='cuda'):
        """
        Args:
            client_data_sizes: 客户端数据大小字典
            cluster_manager: 聚类管理器
            dataset_name: 数据集名称
            data_dir: 数据目录
            device: 设备
        """
        # 导入知识蒸馏聚合器
        from .knowledge_distillation_aggregator import KnowledgeDistillationAggregator
        
        self.kd_aggregator = KnowledgeDistillationAggregator(
            client_data_sizes=client_data_sizes,
            cluster_manager=cluster_manager,
            dataset_name=dataset_name,
            data_dir=data_dir,
            device=device
        )
        
        logging.info("知识蒸馏聚合器包装类初始化完成")
    
    def aggregate(self, shared_states, server_states):
        """执行知识蒸馏聚合"""
        return self.kd_aggregator.aggregate(shared_states, server_states)
    
    def update_distillation_config(self, **kwargs):
        """更新蒸馏配置"""
        self.kd_aggregator.update_distillation_config(**kwargs)
    
    def get_distillation_stats(self):
        """获取蒸馏统计信息"""
        return self.kd_aggregator.distillation_manager.get_data_statistics()