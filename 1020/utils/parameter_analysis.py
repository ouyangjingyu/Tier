import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict

class ClientParameterAnalyzer:
    """客户端模型参数分析器"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.logger = logging.getLogger("ParameterAnalyzer")
    
    def analyze_client_models(self, client_models, save_dir="./visualizations"):
        """
        分析所有客户端模型的参数特征
        
        重点分析：
        1. 分类器权重的类别范数分布
        2. 特征提取层的参数统计
        3. 两者的差异显著性
        """
        self.logger.info("开始分析客户端模型参数...")
        
        # 收集分类器权重范数
        classifier_norms = {}  # {client_id: [norm_class0, norm_class1, ...]}
        
        # 收集特征层参数统计
        feature_layer_stats = {}  # {client_id: {'mean': x, 'std': y, ...}}
        
        for client_id, model in client_models.items():
            # 分析分类器
            classifier_norms[client_id] = self._analyze_classifier(model)
            
            # 分析特征提取层
            feature_layer_stats[client_id] = self._analyze_feature_layers(model)
        
        # 可视化对比
        self._visualize_parameter_differences(
            classifier_norms, 
            feature_layer_stats,
            save_dir
        )
        
        # 计算差异显著性
        self._compute_difference_significance(
            classifier_norms,
            feature_layer_stats
        )
        
        return classifier_norms, feature_layer_stats
    
    def _analyze_classifier(self, model):
        """
        分析分类器层，提取每个类别的权重范数
        
        Returns:
            class_norms: [norm_0, norm_1, ..., norm_9]
        """
        classifier_weight = None
        
        # 找到分类器的权重
        for name, param in model.named_parameters():
            if 'local_classifier' in name and 'weight' in name:
                # 应该是形状 [num_classes, feature_dim]
                if len(param.shape) == 2 and param.shape[0] == self.num_classes:
                    classifier_weight = param.data.cpu().numpy()
                    break
        
        if classifier_weight is None:
            self.logger.warning("未找到分类器权重！")
            return np.zeros(self.num_classes)
        
        # 计算每个类别（每一行）的L2范数
        class_norms = np.linalg.norm(classifier_weight, axis=1)
        
        return class_norms
    
    def _analyze_feature_layers(self, model):
        """
        分析个性化特征提取层的参数统计
        
        Returns:
            stats: 统计特征字典
        """
        all_params = []
        
        for name, param in model.named_parameters():
            # 只收集个性化特征层的参数
            if 'personalized_path' in name:
                param_data = param.data.cpu().numpy().flatten()
                all_params.append(param_data)
        
        if not all_params:
            return {
                'mean': 0.0,
                'std': 0.0,
                'abs_mean': 0.0,
                'max': 0.0
            }
        
        all_params = np.concatenate(all_params)
        
        return {
            'mean': np.mean(all_params),
            'std': np.std(all_params),
            'abs_mean': np.mean(np.abs(all_params)),
            'max': np.max(np.abs(all_params)),
            'num_params': len(all_params)
        }
    
    def _visualize_parameter_differences(self, classifier_norms, 
                                        feature_layer_stats, save_dir):
        """可视化参数差异"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 分类器权重范数热力图
        ax1 = axes[0, 0]
        client_ids = sorted(classifier_norms.keys())
        norm_matrix = np.array([classifier_norms[cid] for cid in client_ids])
        
        sns.heatmap(norm_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(self.num_classes)],
                   yticklabels=[f'Client {cid}' for cid in client_ids],
                   ax=ax1, cbar_kws={'label': 'L2 Norm'})
        ax1.set_title('Classifier Weight Norms per Class\n(Key Differentiator)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Client ID')
        
        # 2. 分类器范数的客户端间差异
        ax2 = axes[0, 1]
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 使用范数向量计算相似度
        similarity_matrix = cosine_similarity(norm_matrix)
        
        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[f'C{cid}' for cid in client_ids],
                   yticklabels=[f'C{cid}' for cid in client_ids],
                   ax=ax2, vmin=0, vmax=1)
        ax2.set_title('Client Similarity\n(Based on Classifier Norms)', 
                     fontsize=12, fontweight='bold')
        
        # 3. 分类器主导类别可视化
        ax3 = axes[1, 0]
        
        for i, cid in enumerate(client_ids):
            norms = classifier_norms[cid]
            top3_classes = np.argsort(norms)[-3:][::-1]
            top3_norms = norms[top3_classes]
            
            ax3.barh([f'Client {cid}'], [1], color='lightgray', alpha=0.3)
            
            # 标注前3个主导类别
            text = f"Top: {top3_classes[0]}({top3_norms[0]:.1f}), " \
                   f"{top3_classes[1]}({top3_norms[1]:.1f}), " \
                   f"{top3_classes[2]}({top3_norms[2]:.1f})"
            ax3.text(0.5, i, text, va='center', ha='center', fontsize=9)
        
        ax3.set_xlim(0, 1)
        ax3.set_title('Dominant Classes per Client\n(Top 3 by Norm)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Normalized Scale')
        
        # 4. 特征层 vs 分类器的变异系数对比
        ax4 = axes[1, 1]
        
        # 计算分类器范数的变异系数（跨客户端）
        classifier_cv = []
        for class_id in range(self.num_classes):
            class_norms_across_clients = [classifier_norms[cid][class_id] 
                                          for cid in client_ids]
            mean = np.mean(class_norms_across_clients)
            std = np.std(class_norms_across_clients)
            cv = std / (mean + 1e-8)
            classifier_cv.append(cv)
        
        avg_classifier_cv = np.mean(classifier_cv)
        
        # 特征层参数的变异系数
        feature_means = [feature_layer_stats[cid]['abs_mean'] for cid in client_ids]
        feature_cv = np.std(feature_means) / (np.mean(feature_means) + 1e-8)
        
        bars = ax4.bar(['Feature Layers', 'Classifier'], 
                      [feature_cv, avg_classifier_cv],
                      color=['skyblue', 'coral'], alpha=0.7)
        
        # 添加数值标注
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Coefficient of Variation (CV)', fontsize=11)
        ax4.set_title('Parameter Variability Across Clients\n(Higher = More Discriminative)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加解释
        if avg_classifier_cv > feature_cv * 1.5:
            conclusion = "Classifier shows significantly higher variability\n→ Better discriminator for clustering"
            color = 'green'
        else:
            conclusion = "Variability is comparable\n→ Both layers contribute"
            color = 'orange'
        
        ax4.text(0.5, 0.95, conclusion,
                transform=ax4.transAxes,
                fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'parameter_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"参数分析可视化已保存至: {save_path}")
    
    def _compute_difference_significance(self, classifier_norms, feature_layer_stats):
        """计算差异显著性"""
        client_ids = sorted(classifier_norms.keys())
        
        # 1. 分类器范数的组间差异
        norm_matrix = np.array([classifier_norms[cid] for cid in client_ids])
        
        # 计算每个类别在不同客户端间的标准差
        class_std = np.std(norm_matrix, axis=0)
        avg_class_std = np.mean(class_std)
        
        # 2. 特征层参数的组间差异
        feature_means = [feature_layer_stats[cid]['abs_mean'] for cid in client_ids]
        feature_std = np.std(feature_means)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("参数差异显著性分析")
        self.logger.info("="*60)
        
        self.logger.info(f"\n分类器权重范数:")
        self.logger.info(f"  平均组间标准差: {avg_class_std:.4f}")
        self.logger.info(f"  最大组间标准差: {np.max(class_std):.4f}")
        self.logger.info(f"  最小组间标准差: {np.min(class_std):.4f}")
        
        self.logger.info(f"\n特征层参数:")
        self.logger.info(f"  组间标准差: {feature_std:.6f}")
        
        ratio = avg_class_std / (feature_std + 1e-8)
        self.logger.info(f"\n差异比率 (分类器/特征层): {ratio:.2f}x")
        
        if ratio > 5:
            self.logger.info("✓ 结论: 分类器权重差异 >> 特征层差异")
            self.logger.info("  推荐: 重点使用分类器权重进行聚类")
        elif ratio > 2:
            self.logger.info("✓ 结论: 分类器权重差异 > 特征层差异")
            self.logger.info("  推荐: 主要使用分类器权重，辅助特征层")
        else:
            self.logger.info("⚠ 结论: 两者差异相当")
            self.logger.info("  推荐: 混合使用两种特征")