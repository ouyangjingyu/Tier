import torch
import copy
import logging
from collections import defaultdict

class SimplifiedGlobalAggregator:
    """简化的全局聚合器，基于客户端数据量进行聚合"""
    
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
        基于数据量权重聚合参数
        
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
        
        # 对每个参数进行加权平均
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
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor
                    else:
                        weighted_sum += weight * param_tensor
            
            # 计算加权平均
            if weighted_sum is not None and total_weight > 0:
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