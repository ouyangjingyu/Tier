import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

class ClientClusterManager:
    """客户端聚类管理器，基于数据分布进行聚类分组"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.client_distributions = {}
        self.cluster_assignments = {}
        self.cluster_info = {}
        self.logger = logging.getLogger("ClientClusterManager")
        
    def calculate_client_distributions(self, train_data_local_dict, y_train, train_net_dataidx_map):
        """计算每个客户端的数据分布"""
        self.logger.info("计算客户端数据分布...")
        
        for client_id, data_indices in train_net_dataidx_map.items():
            # 获取该客户端的标签
            client_labels = y_train[data_indices]
            
            # 计算类别分布
            class_counts = np.zeros(self.num_classes)
            for label in client_labels:
                class_counts[label] += 1
            
            # 计算比例分布
            total_samples = len(client_labels)
            class_proportions = class_counts / total_samples if total_samples > 0 else class_counts
            
            self.client_distributions[client_id] = {
                'class_counts': class_counts,
                'class_proportions': class_proportions,
                'total_samples': total_samples,
                'main_classes': np.where(class_proportions > 0.1)[0].tolist()  # 主要类别
            }
            
            self.logger.info(f"客户端 {client_id} - 总样本: {total_samples}, "
                           f"主要类别: {self.client_distributions[client_id]['main_classes']}, "
                           f"分布: {class_proportions.round(3)}")
    
    def cluster_clients(self, num_clusters=3, method='cosine_similarity'):
        """对客户端进行聚类分组"""
        self.logger.info(f"开始客户端聚类，目标组数: {num_clusters}")
        
        if not self.client_distributions:
            raise ValueError("请先计算客户端数据分布")
        
        client_ids = list(self.client_distributions.keys())
        num_clients = len(client_ids)
        
        # 如果客户端数量小于等于聚类数，每个客户端单独一组
        if num_clients <= num_clusters:
            self.logger.warning(f"客户端数量({num_clients}) <= 聚类数({num_clusters})，每个客户端单独一组")
            for i, client_id in enumerate(client_ids):
                self.cluster_assignments[client_id] = i
            self._analyze_clusters()
            return self.cluster_assignments
        
        # 构建分布矩阵
        distribution_matrix = np.array([
            self.client_distributions[client_id]['class_proportions'] 
            for client_id in client_ids
        ])
        
        if method == 'cosine_similarity':
            # 使用余弦相似度进行聚类
            similarity_matrix = cosine_similarity(distribution_matrix)
            # 转换为距离矩阵
            distance_matrix = 1 - similarity_matrix
            
            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        else:
            # 简单的基于主要类别的分组
            cluster_labels = self._simple_grouping(client_ids, num_clusters)
        
        # 保存聚类结果
        for i, client_id in enumerate(client_ids):
            self.cluster_assignments[client_id] = cluster_labels[i]
        
        # 分析聚类结果
        self._analyze_clusters()
        
        return self.cluster_assignments
    
    def _simple_grouping(self, client_ids, num_clusters):
        """基于主要类别的简单分组"""
        self.logger.info("使用简单主要类别分组方法")
        
        # 收集所有客户端的主要类别
        client_main_classes = {}
        for client_id in client_ids:
            main_classes = self.client_distributions[client_id]['main_classes']
            client_main_classes[client_id] = tuple(sorted(main_classes))
        
        # 按主要类别组合分组
        class_groups = defaultdict(list)
        for client_id, main_classes in client_main_classes.items():
            class_groups[main_classes].append(client_id)
        
        # 如果组数过多，合并相似的组
        unique_groups = list(class_groups.keys())
        if len(unique_groups) > num_clusters:
            # 简单合并：按类别交集大小合并
            merged_groups = self._merge_similar_groups(class_groups, num_clusters)
            class_groups = merged_groups
        
        # 分配聚类标签
        cluster_labels = np.zeros(len(client_ids), dtype=int)
        cluster_id = 0
        
        for group_clients in class_groups.values():
            for client_id in group_clients:
                client_idx = client_ids.index(client_id)
                cluster_labels[client_idx] = cluster_id % num_clusters
            cluster_id += 1
        
        return cluster_labels
    
    def _merge_similar_groups(self, class_groups, target_num_groups):
        """合并相似的类别组"""
        if len(class_groups) <= target_num_groups:
            return class_groups
        
        # 简单合并策略：合并有交集的组
        groups_list = list(class_groups.items())
        merged_groups = {}
        used_indices = set()
        
        merge_id = 0
        for i, (classes1, clients1) in enumerate(groups_list):
            if i in used_indices:
                continue
                
            merged_clients = clients1.copy()
            used_indices.add(i)
            
            # 寻找有交集的组进行合并
            for j, (classes2, clients2) in enumerate(groups_list[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # 计算类别交集
                intersection = set(classes1) & set(classes2)
                if intersection and len(merged_groups) < target_num_groups - 1:
                    merged_clients.extend(clients2)
                    used_indices.add(j)
            
            merged_groups[f"group_{merge_id}"] = merged_clients
            merge_id += 1
            
            if len(merged_groups) >= target_num_groups:
                break
        
        # 处理剩余的客户端
        remaining_clients = []
        for i, (_, clients) in enumerate(groups_list):
            if i not in used_indices:
                remaining_clients.extend(clients)
        
        if remaining_clients and len(merged_groups) < target_num_groups:
            merged_groups[f"group_{merge_id}"] = remaining_clients
        elif remaining_clients:
            # 将剩余客户端分配到现有组
            group_keys = list(merged_groups.keys())
            for i, client in enumerate(remaining_clients):
                group_key = group_keys[i % len(group_keys)]
                merged_groups[group_key].append(client)
        
        return merged_groups
    
    def _analyze_clusters(self):
        """分析聚类结果"""
        self.logger.info("分析聚类结果...")
        
        # 按组组织客户端
        groups = defaultdict(list)
        for client_id, cluster_id in self.cluster_assignments.items():
            groups[cluster_id].append(client_id)
        
        # 分析每个组
        for cluster_id, client_list in groups.items():
            group_info = {
                'clients': client_list,
                'size': len(client_list),
                'total_samples': 0,
                'class_distribution': np.zeros(self.num_classes),
                'avg_similarity': 0.0
            }
            
            # 计算组内统计信息
            distributions = []
            total_samples = 0
            
            for client_id in client_list:
                client_info = self.client_distributions[client_id]
                total_samples += client_info['total_samples']
                group_info['class_distribution'] += client_info['class_counts']
                distributions.append(client_info['class_proportions'])
            
            group_info['total_samples'] = total_samples
            
            # 计算组内平均相似性
            if len(distributions) > 1:
                similarity_matrix = cosine_similarity(distributions)
                # 计算上三角矩阵的平均值（排除对角线）
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                group_info['avg_similarity'] = similarity_matrix[mask].mean()
            else:
                group_info['avg_similarity'] = 1.0
            
            # 计算组的类别分布比例
            if total_samples > 0:
                group_info['class_proportions'] = group_info['class_distribution'] / total_samples
            else:
                group_info['class_proportions'] = np.zeros(self.num_classes)
            
            self.cluster_info[cluster_id] = group_info
            
            self.logger.info(f"组 {cluster_id}: 客户端 {client_list}, "
                           f"样本数: {total_samples}, "
                           f"组内相似性: {group_info['avg_similarity']:.3f}, "
                           f"主要类别: {np.where(group_info['class_proportions'] > 0.1)[0].tolist()}")
    
    def get_cluster_assignments(self):
        """获取聚类分配结果"""
        return self.cluster_assignments
    
    def get_cluster_info(self):
        """获取聚类信息"""
        return self.cluster_info
    
    def get_clients_in_cluster(self, cluster_id):
        """获取指定组中的客户端列表"""
        return [client_id for client_id, cid in self.cluster_assignments.items() if cid == cluster_id]
    
    def get_cluster_data_distribution(self, cluster_id):
        """获取指定组的数据分布"""
        if cluster_id in self.cluster_info:
            return self.cluster_info[cluster_id]['class_proportions']
        return None
    
    def calculate_cluster_coverage_weights(self):
        """计算各组的覆盖度权重（用于二级聚合）"""
        coverage_weights = {}
        
        for cluster_id, info in self.cluster_info.items():
            proportions = info['class_proportions']
            # 计算Shannon熵作为覆盖度度量
            # 避免log(0)
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