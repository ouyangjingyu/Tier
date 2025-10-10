import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import logging

class DataDistributionVisualizer:
    """数据分布可视化工具"""
    
    def __init__(self, save_dir="./visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        
    def visualize_client_data_distribution(self, y_train, net_dataidx_map, 
                                         num_classes=10, dataset_name="Dataset"):
        """可视化客户端数据分布"""
        logging.info(f"生成{dataset_name}客户端数据分布可视化图...")
        
        # 计算每个客户端的类别分布
        client_distributions = {}
        for client_id, data_indices in net_dataidx_map.items():
            class_counts = np.zeros(num_classes)
            if len(data_indices) > 0:
                client_labels = y_train[data_indices]
                for label in client_labels:
                    class_counts[label] += 1
            client_distributions[client_id] = class_counts
        
        # 创建分布矩阵
        num_clients = len(client_distributions)
        distribution_matrix = np.zeros((num_clients, num_classes))
        
        for i, (client_id, counts) in enumerate(client_distributions.items()):
            total_samples = np.sum(counts)
            if total_samples > 0:
                distribution_matrix[i] = counts / total_samples
            else:
                distribution_matrix[i] = counts
        
        # 1. 热力图
        self._plot_heatmap(distribution_matrix, dataset_name, num_clients, num_classes)
        
        # 2. 客户端分布柱状图
        self._plot_client_distributions(client_distributions, dataset_name, num_classes)
        
        # 3. 异质性统计图
        self._plot_heterogeneity_stats(distribution_matrix, dataset_name)
        
        # 4. 聚类可视化（如果有聚类结果）
        return distribution_matrix
        
    def _plot_heatmap(self, distribution_matrix, dataset_name, num_clients, num_classes):
        """绘制客户端-类别分布热力图"""
        plt.figure(figsize=(12, 8))
        
        # 创建热力图
        ax = sns.heatmap(distribution_matrix, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='YlOrRd',
                        xticklabels=[f'Class {i}' for i in range(num_classes)],
                        yticklabels=[f'Client {i}' for i in range(num_clients)],
                        cbar_kws={'label': 'Proportion'})
        
        plt.title(f'{dataset_name} - Client Data Distribution Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Clients', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'{dataset_name.lower()}_distribution_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"热力图已保存至: {save_path}")
    
    def _plot_client_distributions(self, client_distributions, dataset_name, num_classes):
        """绘制每个客户端的类别分布柱状图"""
        num_clients = len(client_distributions)
        
        # 计算子图布局
        cols = min(4, num_clients)
        rows = (num_clients + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_clients == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # 为每个客户端绘制分布图
        for i, (client_id, counts) in enumerate(client_distributions.items()):
            row, col = i // cols, i % cols
            
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            else:
                ax = axes[row, col]
            
            total_samples = np.sum(counts)
            proportions = counts / total_samples if total_samples > 0 else counts
            
            # 绘制柱状图
            bars = ax.bar(range(num_classes), proportions, 
                         color=plt.cm.Set3(i/num_clients), alpha=0.8)
            
            ax.set_title(f'Client {client_id}\n({int(total_samples)} samples)', fontsize=10)
            ax.set_xlabel('Classes', fontsize=9)
            ax.set_ylabel('Proportion', fontsize=9)
            ax.set_xticks(range(num_classes))
            ax.set_ylim(0, 1.0)
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # 只显示大于1%的标签
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的子图
        for i in range(num_clients, rows * cols):
            row, col = i // cols, i % cols
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            else:
                ax = axes[row, col]
            ax.set_visible(False)
        
        plt.suptitle(f'{dataset_name} - Individual Client Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'{dataset_name.lower()}_client_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"客户端分布图已保存至: {save_path}")
    
    def _plot_heterogeneity_stats(self, distribution_matrix, dataset_name):
        """绘制异质性统计图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 每个客户端的基尼系数（衡量数据不平衡程度）
        gini_coefficients = []
        for i in range(distribution_matrix.shape[0]):
            dist = distribution_matrix[i]
            # 计算基尼系数
            sorted_dist = np.sort(dist)
            n = len(sorted_dist)
            cumsum_dist = np.cumsum(sorted_dist)
            gini = (n + 1 - 2 * np.sum(cumsum_dist)) / (n * np.sum(sorted_dist)) if np.sum(sorted_dist) > 0 else 0
            gini_coefficients.append(gini)
        
        axes[0].bar(range(len(gini_coefficients)), gini_coefficients, color='skyblue', alpha=0.7)
        axes[0].set_title('Client Data Imbalance (Gini Coefficient)', fontweight='bold')
        axes[0].set_xlabel('Client ID')
        axes[0].set_ylabel('Gini Coefficient')
        axes[0].set_ylim(0, 1)
        
        # 2. 每个类别在所有客户端中的分布
        class_distributions = distribution_matrix.T  # 转置得到类别视角
        
        box_data = []
        for class_id in range(class_distributions.shape[0]):
            class_dist = class_distributions[class_id]
            box_data.append(class_dist[class_dist > 0])  # 只包含非零值
        
        axes[1].boxplot(box_data, labels=[f'C{i}' for i in range(len(box_data))])
        axes[1].set_title('Class Distribution Across Clients', fontweight='bold')
        axes[1].set_xlabel('Class ID')
        axes[1].set_ylabel('Proportion')
        
        # 3. 客户端间相似性矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(distribution_matrix)
        
        im = axes[2].imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Client Similarity Matrix', fontweight='bold')
        axes[2].set_xlabel('Client ID')
        axes[2].set_ylabel('Client ID')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[2])
        cbar.set_label('Cosine Similarity')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'{dataset_name.lower()}_heterogeneity_stats.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"异质性统计图已保存至: {save_path}")
        
        # 输出统计信息
        avg_gini = np.mean(gini_coefficients)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
        
        logging.info(f"{dataset_name} 异质性统计:")
        logging.info(f"  平均基尼系数: {avg_gini:.3f} (越高越不平衡)")
        logging.info(f"  平均客户端相似性: {avg_similarity:.3f} (越低异质性越强)")
        
    def visualize_clustering_results(self, distribution_matrix, cluster_assignments, dataset_name):
        """可视化聚类结果"""
        if not cluster_assignments:
            return
            
        logging.info(f"生成{dataset_name}聚类结果可视化...")
        
        # 按聚类重新排序分布矩阵
        sorted_indices = sorted(range(len(cluster_assignments)), 
                              key=lambda x: cluster_assignments[x])
        sorted_matrix = distribution_matrix[sorted_indices]
        
        # 创建聚类标签
        cluster_labels = [f'Client {sorted_indices[i]}\n(Group {cluster_assignments[sorted_indices[i]]})' 
                         for i in range(len(sorted_indices))]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制重新排序的热力图
        ax = sns.heatmap(sorted_matrix,
                        annot=True,
                        fmt='.2f',
                        cmap='YlOrRd',
                        xticklabels=[f'Class {i}' for i in range(sorted_matrix.shape[1])],
                        yticklabels=cluster_labels,
                        cbar_kws={'label': 'Proportion'})
        
        # 添加聚类分隔线
        cluster_boundaries = []
        current_cluster = cluster_assignments[sorted_indices[0]]
        for i, idx in enumerate(sorted_indices):
            if cluster_assignments[idx] != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster_assignments[idx]
        
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary, color='red', linewidth=2, linestyle='--')
        
        plt.title(f'{dataset_name} - Clustered Client Data Distribution', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Clients (Grouped)', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'{dataset_name.lower()}_clustering_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"聚类结果图已保存至: {save_path}")