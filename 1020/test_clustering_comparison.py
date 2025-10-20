"""
客户端聚类方案对比测试脚本（完整版）
支持：
1. 基础对比测试
2. 参数分析
3. 消融实验
"""

import torch
import numpy as np
import os
import sys
import logging
import argparse
import warnings
import time

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入必要的模块
from model.resnet import TierAwareClientModel
from utils.softlabel_clustering import SoftLabelClusterManager, compare_clustering_results
from utils.client_clustering import ClientClusterManager
from utils.data_visualization import DataDistributionVisualizer
from utils.distillation_data_manager import DistillationDataManager
from utils.parameter_analysis import ClientParameterAnalyzer

# 数据加载
from api.data_preprocessing.cifar10.data_loader import partition_data as partition_cifar10
from api.data_preprocessing.fashion_mnist.data_loader import partition_data as partition_fashion_mnist

warnings.filterwarnings("ignore")

def setup_logging(log_filename="clustering_comparison_test.log"):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger("ClusteringTest")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='客户端聚类方案对比测试（完整版）')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                       choices=['cifar10', 'fashion_mnist'],
                       help='数据集')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero',
                       help='数据划分方法')
    parser.add_argument('--partition_alpha', type=float, default=0.5,
                       help='数据异质性参数')
    parser.add_argument('--client_number', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--num_clusters', type=int, default=3,
                       help='聚类数量')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    
    # 模型配置
    parser.add_argument('--model_type', type=str, default='resnet56',
                       help='模型类型')
    
    # 训练配置
    parser.add_argument('--warmup_epochs', type=int, default=15,
                       help='预热训练轮数（建议10-20）')
    parser.add_argument('--warmup_lr', type=float, default=0.01,
                       help='预热学习率')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='早停耐心值')
    
    # 聚类方法配置
    parser.add_argument('--use_local_testset', action='store_true',
                       help='使用本地测试集而非蒸馏数据集（推荐）')
    parser.add_argument('--clustering_method', type=str, default='cosine_similarity',
                       choices=['cosine_similarity', 'euclidean'],
                       help='聚类相似度度量方法')
    
    # 实验模式
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'parameter_analysis', 'ablation', 'full'],
                       help='实验模式：basic=基础对比, parameter_analysis=参数分析, '
                            'ablation=消融实验, full=完整测试')
    
    # 输出配置
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='可视化结果保存目录')
    
    return parser.parse_args()

def load_data_and_partition(args, logger):
    """
    加载并划分数据集
    
    Returns:
        数据划分结果和数据加载器
    """
    logger.info(f"加载数据集: {args.dataset}")
    
    # 根据数据集选择划分函数
    if args.dataset == 'cifar10':
        partition_func = partition_cifar10
        input_channels = 3
        num_classes = 10
    elif args.dataset == 'fashion_mnist':
        partition_func = partition_fashion_mnist
        input_channels = 1
        num_classes = 10
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 划分数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_func(
            args.dataset, 
            args.data_dir, 
            args.partition_method,
            args.client_number, 
            args.partition_alpha
        )
    
    logger.info(f"数据划分完成，客户端数量: {args.client_number}")
    
    # 创建数据加载器
    from api.data_preprocessing.cifar10.data_loader import get_dataloader_CIFAR10
    from api.data_preprocessing.fashion_mnist.data_loader import get_dataloader
    
    train_data_local_dict = {}
    test_data_local_dict = {}
    
    for client_id in range(args.client_number):
        train_dataidxs = train_net_dataidx_map[client_id]
        test_dataidxs = test_net_dataidx_map[client_id]
        
        if args.dataset == 'cifar10':
            train_data_local, test_data_local = get_dataloader_CIFAR10(
                args.data_dir, args.batch_size, args.batch_size,
                train_dataidxs, test_dataidxs
            )
        else:  # fashion_mnist
            train_data_local, test_data_local = get_dataloader(
                args.dataset, args.data_dir, args.batch_size, args.batch_size,
                train_dataidxs, test_dataidxs
            )
        
        train_data_local_dict[client_id] = train_data_local
        test_data_local_dict[client_id] = test_data_local
        
        logger.info(f"客户端 {client_id} - 训练样本: {len(train_dataidxs)}, "
                   f"测试样本: {len(test_dataidxs)}")
    
    return {
        'y_train': y_train,
        'train_net_dataidx_map': train_net_dataidx_map,
        'test_net_dataidx_map': test_net_dataidx_map,
        'train_data_local_dict': train_data_local_dict,
        'test_data_local_dict': test_data_local_dict,
        'input_channels': input_channels,
        'num_classes': num_classes
    }

def original_clustering_method(data_info, args, logger):
    """
    原方案：基于数据分布的聚类
    
    Returns:
        cluster_assignments: {client_id: cluster_id}
        cluster_manager: ClientClusterManager实例
    """
    logger.info("\n" + "="*60)
    logger.info("执行原方案：基于数据分布的聚类")
    logger.info("="*60)
    
    # 创建聚类管理器
    cluster_manager = ClientClusterManager(num_classes=data_info['num_classes'])
    
    # 计算客户端数据分布
    cluster_manager.calculate_client_distributions(
        data_info['train_data_local_dict'],
        data_info['y_train'],
        data_info['train_net_dataidx_map']
    )
    
    # 执行聚类
    cluster_assignments = cluster_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method='cosine_similarity'
    )
    
    logger.info(f"原方案聚类结果: {cluster_assignments}")
    
    return cluster_assignments, cluster_manager

def prepare_client_models_and_softlabels(data_info, args, logger):
    """
    准备客户端模型并进行预热训练，收集软标签
    这是所有新方案的共同准备步骤
    
    Returns:
        client_models: 训练好的客户端模型
        softlabel_manager: 软标签聚类管理器
        distillation_loader: 蒸馏数据加载器（如果使用）
    """
    logger.info("\n" + "="*60)
    logger.info("准备客户端模型和软标签")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 1. 创建客户端模型
    logger.info("\n创建客户端模型...")
    client_models = {}
    for client_id in range(args.client_number):
        client_models[client_id] = TierAwareClientModel(
            num_classes=data_info['num_classes'],
            tier=2,  # 使用中等tier
            model_type=args.model_type,
            input_channels=data_info['input_channels']
        )
    
    # 2. 创建软标签聚类管理器
    softlabel_manager = SoftLabelClusterManager(
        num_classes=data_info['num_classes'],
        device=device
    )
    
    # 3. 预热训练
    logger.info(f"\n开始预热训练，轮数: {args.warmup_epochs}")
    start_time = time.time()
    
    client_models = softlabel_manager.warmup_train_clients(
        client_models=client_models,
        train_data_local_dict=data_info['train_data_local_dict'],
        test_data_local_dict=data_info['test_data_local_dict'],
        warmup_epochs=args.warmup_epochs,
        lr=args.warmup_lr,
        early_stop_patience=args.early_stop_patience
    )
    
    warmup_time = time.time() - start_time
    logger.info(f"预热训练完成，耗时: {warmup_time:.2f}秒")
    
    # 4. 收集软标签
    distillation_loader = None
    
    if args.use_local_testset:
        logger.info("\n使用本地测试集收集软标签...")
        softlabel_manager.collect_soft_labels_from_local_testset(
            client_models=client_models,
            test_data_local_dict=data_info['test_data_local_dict']
        )
    else:
        logger.info("\n使用蒸馏数据集收集软标签...")
        distillation_manager = DistillationDataManager(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            distillation_ratio=0.15,
            batch_size=args.batch_size
        )
        distillation_loader, _ = distillation_manager.create_distillation_dataset()
        
        softlabel_manager.collect_soft_labels(
            client_models=client_models,
            distillation_loader=distillation_loader
        )
    
    logger.info("软标签收集完成")
    
    return client_models, softlabel_manager, distillation_loader

def run_parameter_analysis(client_models, data_info, args, logger):
    """
    运行参数分析实验
    分析分类器权重 vs 特征层参数的差异显著性
    """
    logger.info("\n" + "="*60)
    logger.info("参数分析实验")
    logger.info("="*60)
    
    # 创建参数分析器
    analyzer = ClientParameterAnalyzer(num_classes=data_info['num_classes'])
    
    # 分析所有客户端模型
    classifier_norms, feature_layer_stats = analyzer.analyze_client_models(
        client_models=client_models,
        save_dir=args.save_dir
    )
    
    logger.info("\n参数分析完成，可视化结果已保存")
    
    return classifier_norms, feature_layer_stats

def run_ablation_study(client_models, softlabel_manager, original_assignments, 
                      data_info, args, logger):
    """
    运行消融实验
    对比不同特征组合的聚类效果
    """
    logger.info("\n" + "="*60)
    logger.info("消融实验：不同特征组合对比")
    logger.info("="*60)
    
    results = {}
    all_assignments = {}
    
    # 方案1: 仅软标签统计特征
    logger.info("\n【方案1】仅软标签统计特征")
    logger.info("-" * 60)
    
    softlabel_manager.client_statistics = {}
    softlabel_manager.extract_statistical_features()
    assignments_v1 = softlabel_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method=args.clustering_method
    )
    metrics_v1 = compare_clustering_results(original_assignments, assignments_v1)
    results['仅软标签'] = metrics_v1
    all_assignments['仅软标签'] = assignments_v1
    
    logger.info(f"聚类结果: {assignments_v1}")
    logger.info(f"NMI: {metrics_v1['NMI']:.4f}, "
               f"ARI: {metrics_v1['ARI']:.4f}, "
               f"Accuracy: {metrics_v1['Accuracy']:.2%}")
    
    # 方案2: 软标签 + 分类器权重（推荐）
    logger.info("\n【方案2】软标签 + 分类器权重范数（推荐）")
    logger.info("-" * 60)
    
    softlabel_manager.client_statistics = {}
    softlabel_manager.extract_classifier_focused_features(client_models)
    assignments_v2 = softlabel_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method=args.clustering_method
    )
    metrics_v2 = compare_clustering_results(original_assignments, assignments_v2)
    results['软标签+分类器'] = metrics_v2
    all_assignments['软标签+分类器'] = assignments_v2
    
    logger.info(f"聚类结果: {assignments_v2}")
    logger.info(f"NMI: {metrics_v2['NMI']:.4f}, "
               f"ARI: {metrics_v2['ARI']:.4f}, "
               f"Accuracy: {metrics_v2['Accuracy']:.2%}")
    
    # 方案3: 完整混合特征（软标签 + 分类器 + 特征层）
    logger.info("\n【方案3】完整混合特征（软标签 + 分类器 + 特征层）")
    logger.info("-" * 60)
    
    softlabel_manager.client_statistics = {}
    softlabel_manager.extract_improved_hybrid_features(client_models)
    assignments_v3 = softlabel_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method=args.clustering_method
    )
    metrics_v3 = compare_clustering_results(original_assignments, assignments_v3)
    results['完整混合'] = metrics_v3
    all_assignments['完整混合'] = assignments_v3
    
    logger.info(f"聚类结果: {assignments_v3}")
    logger.info(f"NMI: {metrics_v3['NMI']:.4f}, "
               f"ARI: {metrics_v3['ARI']:.4f}, "
               f"Accuracy: {metrics_v3['Accuracy']:.2%}")
    
    # 打印对比总结
    logger.info("\n" + "="*60)
    logger.info("消融实验结果汇总")
    logger.info("="*60)
    logger.info(f"\n{'方案':<20} {'NMI':<12} {'ARI':<12} {'Accuracy':<12}")
    logger.info("-" * 60)
    
    for name, metrics in results.items():
        logger.info(
            f"{name:<20} "
            f"{metrics['NMI']:<12.4f} "
            f"{metrics['ARI']:<12.4f} "
            f"{metrics['Accuracy']:<12.2%}"
        )
    
    # 找出最佳方案
    best_method_name = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
    best_metrics = results[best_method_name]
    
    logger.info("\n" + "="*60)
    logger.info(f"✓ 最佳方案: {best_method_name}")
    logger.info(f"  NMI: {best_metrics['NMI']:.4f}")
    logger.info(f"  ARI: {best_metrics['ARI']:.4f}")
    logger.info(f"  准确率: {best_metrics['Accuracy']:.2%}")
    logger.info("="*60)
    
    # 可视化消融实验结果
    visualize_ablation_results(results, all_assignments, original_assignments,
                               data_info, args, logger)
    
    return results, all_assignments, best_method_name

def visualize_ablation_results(results, all_assignments, original_assignments,
                               data_info, args, logger):
    """可视化消融实验结果"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("\n生成消融实验可视化...")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 指标对比柱状图
    ax1 = axes[0, 0]
    method_names = list(results.keys())
    nmis = [results[m]['NMI'] for m in method_names]
    aris = [results[m]['ARI'] for m in method_names]
    accs = [results[m]['Accuracy'] for m in method_names]
    
    x = np.arange(len(method_names))
    width = 0.25
    
    ax1.bar(x - width, nmis, width, label='NMI', alpha=0.8)
    ax1.bar(x, aris, width, label='ARI', alpha=0.8)
    ax1.bar(x + width, accs, width, label='Accuracy', alpha=0.8)
    
    ax1.set_xlabel('方案', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('消融实验 - 指标对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 聚类一致性对比
    ax2 = axes[0, 1]
    
    client_ids = sorted(original_assignments.keys())
    n_clients = len(client_ids)
    n_methods = len(method_names)
    
    consistency_matrix = np.zeros((n_clients, n_methods))
    
    for i, cid in enumerate(client_ids):
        orig_cluster = original_assignments[cid]
        for j, method in enumerate(method_names):
            pred_cluster = all_assignments[method][cid]
            consistency_matrix[i, j] = 1 if orig_cluster == pred_cluster else 0
    
    sns.heatmap(consistency_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
               xticklabels=method_names, 
               yticklabels=[f'C{cid}' for cid in client_ids],
               ax=ax2, vmin=0, vmax=1, cbar_kws={'label': '一致性'})
    ax2.set_title('客户端聚类一致性\n(1=与原方案一致, 0=不一致)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('方案')
    ax2.set_ylabel('客户端ID')
    
    # 3. 准确率提升对比
    ax3 = axes[1, 0]
    
    baseline_acc = results[method_names[0]]['Accuracy']
    improvements = [(results[m]['Accuracy'] - baseline_acc) * 100 
                   for m in method_names]
    
    colors = ['gray' if imp == 0 else ('green' if imp > 0 else 'red') 
             for imp in improvements]
    bars = ax3.barh(method_names, improvements, color=colors, alpha=0.7)
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('准确率提升 (%)', fontsize=11)
    ax3.set_title(f'相对基线方案的准确率提升\n(基线: {method_names[0]})', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标注
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        if imp >= 0:
            ax3.text(imp + 0.5, i, f'+{imp:.1f}%', va='center', fontsize=10)
        else:
            ax3.text(imp - 0.5, i, f'{imp:.1f}%', va='center', ha='right', fontsize=10)
    
    # 4. 详细对比表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    headers = ['方案', 'NMI', 'ARI', 'Accuracy', '相对提升']
    
    for method in method_names:
        metrics = results[method]
        improvement = (metrics['Accuracy'] - baseline_acc) * 100
        row = [
            method,
            f"{metrics['NMI']:.3f}",
            f"{metrics['ARI']:.3f}",
            f"{metrics['Accuracy']:.2%}",
            f"{improvement:+.1f}%" if improvement != 0 else "-"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 高亮最佳结果
    best_method_idx = method_names.index(
        max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
    )
    for col in range(len(headers)):
        table[(best_method_idx + 1, col)].set_facecolor('#90EE90')
    
    ax4.set_title('详细指标对比表', fontsize=12, fontweight='bold', pad=20)
    
    # 总标题
    fig.suptitle(
        f'{args.dataset.upper()} - 消融实验结果\n'
        f'客户端数: {args.client_number}, 聚类数: {args.num_clusters}, '
        f'预热轮数: {args.warmup_epochs}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 
                            f'{args.dataset}_ablation_study.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"消融实验可视化已保存至: {save_path}")

def run_basic_comparison(client_models, softlabel_manager, original_assignments,
                        data_info, args, logger):
    """
    运行基础对比测试
    对比原方案 vs 最佳新方案
    """
    logger.info("\n" + "="*60)
    logger.info("基础对比测试")
    logger.info("="*60)
    
    # 使用推荐的方案：软标签 + 分类器权重
    logger.info("\n使用推荐方案：软标签 + 分类器权重")
    
    softlabel_manager.extract_classifier_focused_features(client_models)
    softlabel_assignments = softlabel_manager.cluster_clients(
        num_clusters=args.num_clusters,
        method=args.clustering_method
    )
    
    logger.info(f"新方案聚类结果: {softlabel_assignments}")
    
    # 对比和可视化
    metrics = compare_and_visualize(
        original_assignments=original_assignments,
        softlabel_assignments=softlabel_assignments,
        data_info=data_info,
        args=args,
        logger=logger
    )
    
    return softlabel_assignments, metrics

def compare_and_visualize(original_assignments, softlabel_assignments,
                         data_info, args, logger):
    """
    对比两种聚类方案并可视化
    """
    logger.info("\n" + "="*60)
    logger.info("对比两种聚类方案")
    logger.info("="*60)
    
    # 1. 计算对比指标
    metrics = compare_clustering_results(
        true_assignments=original_assignments,
        predicted_assignments=softlabel_assignments
    )
    
    logger.info("\n聚类质量指标:")
    logger.info(f"  NMI (归一化互信息): {metrics['NMI']:.4f}")
    logger.info(f"  ARI (调整兰德指数): {metrics['ARI']:.4f}")
    logger.info(f"  精确匹配率: {metrics['Accuracy']:.2%}")
    
    # 2. 详细对比
    logger.info("\n客户端分配详细对比:")
    logger.info(f"{'客户端ID':<10} {'原方案':<10} {'新方案':<10} {'是否一致':<10}")
    logger.info("-" * 40)
    
    match_count = 0
    for client_id in sorted(original_assignments.keys()):
        orig = original_assignments[client_id]
        soft = softlabel_assignments[client_id]
        match = "✓" if orig == soft else "✗"
        if orig == soft:
            match_count += 1
        
        logger.info(f"{client_id:<10} {orig:<10} {soft:<10} {match:<10}")
    
    logger.info(f"\n直接匹配数: {match_count}/{len(original_assignments)}")
    
    # 3. 计算数据分布矩阵
    logger.info("\n计算数据分布矩阵用于可视化...")
    distribution_matrix = np.zeros((args.client_number, data_info['num_classes']))
    
    for client_id in range(args.client_number):
        client_indices = data_info['train_net_dataidx_map'][client_id]
        client_labels = data_info['y_train'][client_indices]
        
        for class_id in range(data_info['num_classes']):
            count = np.sum(client_labels == class_id)
            if len(client_indices) > 0:
                distribution_matrix[client_id, class_id] = count / len(client_indices)
    
    # 4. 可视化对比
    logger.info("\n生成可视化对比图...")
    visualizer = DataDistributionVisualizer(save_dir=args.save_dir)
    
    visualizer.visualize_clustering_comparison(
        distribution_matrix=distribution_matrix,
        original_assignments=original_assignments,
        softlabel_assignments=softlabel_assignments,
        dataset_name=args.dataset.upper(),
        comparison_metrics=metrics
    )
    
    logger.info(f"可视化完成，图片已保存至 {args.save_dir}/")
    
    return metrics

def print_summary(args, results, logger):
    """打印测试总结"""
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    logger.info(f"\n数据集配置:")
    logger.info(f"  数据集: {args.dataset}")
    logger.info(f"  客户端数量: {args.client_number}")
    logger.info(f"  聚类数量: {args.num_clusters}")
    logger.info(f"  异质性参数α: {args.partition_alpha}")
    logger.info(f"  预热轮数: {args.warmup_epochs}")
    logger.info(f"  使用本地测试集: {'是' if args.use_local_testset else '否'}")
    
    if args.mode == 'basic':
        metrics = results
        logger.info(f"\n聚类性能对比:")
        logger.info(f"  归一化互信息 (NMI): {metrics['NMI']:.4f} (越接近1越好)")
        logger.info(f"  调整兰德指数 (ARI): {metrics['ARI']:.4f} (越接近1越好)")
        logger.info(f"  精确匹配率: {metrics['Accuracy']:.2%}")
        
        # 性能评价
        if metrics['Accuracy'] >= 0.95:
            performance = "优秀 - 与原方案几乎完全一致"
        elif metrics['Accuracy'] >= 0.85:
            performance = "良好 - 与原方案基本一致"
        elif metrics['Accuracy'] >= 0.70:
            performance = "中等 - 部分客户端分组不同"
        else:
            performance = "较差 - 需要调整参数或方法"
        
        logger.info(f"\n总体评价: {performance}")
        
    elif args.mode == 'ablation':
        logger.info(f"\n消融实验结果:")
        best_method = max(results.items(), key=lambda x: x[1]['Accuracy'])
        logger.info(f"  最佳方案: {best_method[0]}")
        logger.info(f"  准确率: {best_method[1]['Accuracy']:.2%}")
        logger.info(f"  NMI: {best_method[1]['NMI']:.4f}")
        logger.info(f"  ARI: {best_method[1]['ARI']:.4f}")

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 配置日志
    log_filename = f"clustering_test_{args.mode}_{args.dataset}.log"
    logger = setup_logging(log_filename)
    
    logger.info("="*60)
    logger.info(f"客户端聚类方案对比测试 - {args.mode.upper()}模式")
    logger.info("="*60)
    
    try:
        # 1. 加载并划分数据
        data_info = load_data_and_partition(args, logger)
        
        # 2. 原方案聚类
        original_assignments, original_manager = original_clustering_method(
            data_info, args, logger
        )
        
        # 3. 准备客户端模型和软标签（所有新方案的共同步骤）
        client_models, softlabel_manager, distillation_loader = \
            prepare_client_models_and_softlabels(data_info, args, logger)
        
        # 4. 根据模式执行不同的实验
        if args.mode == 'parameter_analysis':
            # 参数分析模式
            run_parameter_analysis(client_models, data_info, args, logger)
            logger.info("\n参数分析完成！请查看生成的可视化图表。")
            
        elif args.mode == 'ablation':
            # 消融实验模式
            results, all_assignments, best_method = run_ablation_study(
                client_models, softlabel_manager, original_assignments,
                data_info, args, logger
            )
            
            print_summary(args, results, logger)
            
        elif args.mode == 'basic':
            # 基础对比模式
            softlabel_assignments, metrics = run_basic_comparison(
                client_models, softlabel_manager, original_assignments,
                data_info, args, logger
            )
            
            print_summary(args, metrics, logger)
            
        elif args.mode == 'full':
            # 完整测试模式：依次执行所有实验
            logger.info("\n开始完整测试流程...")
            
            # 步骤1: 参数分析
            logger.info("\n【步骤1/3】参数分析")
            run_parameter_analysis(client_models, data_info, args, logger)
            
            # 步骤2: 消融实验
            logger.info("\n【步骤2/3】消融实验")
            results, all_assignments, best_method = run_ablation_study(
                client_models, softlabel_manager, original_assignments,
                data_info, args, logger
            )
            
            # 步骤3: 使用最佳方案进行详细对比
            logger.info("\n【步骤3/3】使用最佳方案进行详细对比")
            
            # 重置统计信息
            softlabel_manager.client_statistics = {}
            
            # 根据最佳方案提取特征
            if best_method == '软标签+分类器':
                softlabel_manager.extract_classifier_focused_features(client_models)
            elif best_method == '完整混合':
                softlabel_manager.extract_improved_hybrid_features(client_models)
            else:
                softlabel_manager.extract_statistical_features()
            
            best_assignments = softlabel_manager.cluster_clients(
                num_clusters=args.num_clusters,
                method=args.clustering_method
            )
            
            # 生成详细对比可视化
            metrics = compare_and_visualize(
                original_assignments=original_assignments,
                softlabel_assignments=best_assignments,
                data_info=data_info,
                args=args,
                logger=logger
            )
            
            logger.info(f"\n完整测试完成！最佳方案: {best_method}")
            print_summary(args, results, logger)
        
        logger.info("\n" + "="*60)
        logger.info("测试完成！")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()