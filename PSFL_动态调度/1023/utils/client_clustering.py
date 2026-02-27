import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# 可以大幅简化这个文件，只保留必要的接口
class ClientClusterManager:
    """客户端聚类管理器 - 兼容层"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.client_distributions = {}
        self.cluster_assignments = {}
        self.cluster_info = {}
        self.logger = logging.getLogger("ClientClusterManager")
    
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
    
    def calculate_cluster_coverage_weights(self):
        """计算各组的覆盖度权重（用于二级聚合）"""
        coverage_weights = {}
        
        for cluster_id, info in self.cluster_info.items():
            proportions = info.get('avg_predicted_distribution', 
                                  info.get('class_proportions', np.ones(self.num_classes)/self.num_classes))
            
            # 计算Shannon熵作为覆盖度度量
            proportions_safe = proportions + 1e-8
            entropy = -np.sum(proportions_safe * np.log(proportions_safe))
            coverage_weights[cluster_id] = entropy
        
        # 归一化权重
        total_coverage = sum(coverage_weights.values())
        if total_coverage > 0:
            for cluster_id in coverage_weights:
                coverage_weights[cluster_id] /= total_coverage
        
        self.logger.info(f"组覆盖度权重: {coverage_weights}")
        return coverage_weights