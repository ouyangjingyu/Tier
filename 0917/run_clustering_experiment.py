#!/usr/bin/env python3
"""
聚类实验运行脚本
比较传统聚合方法和分层聚合方法的性能
支持Fashion-MNIST和CIFAR-10数据集
"""

import subprocess
import sys
import os

def run_experiment(experiment_name, dataset="fashion_mnist", enable_clustering=False, 
                  num_clusters=3, clustering_method='cosine_similarity', rounds=50):
    """运行单个实验"""
    
    cmd = [
        sys.executable, "main.py",
        "--running_name", experiment_name,
        "--dataset", dataset,
        "--client_number", "10",
        "--rounds", str(rounds),
        "--batch_size", "256",
        "--lr", "0.005",
        "--partition_method", "hetero",
        "--partition_alpha", "0.5"
    ]
    
    if enable_clustering:
        cmd.extend([
            "--enable_clustering",
            "--num_clusters", str(num_clusters),
            "--clustering_method", clustering_method,
            "--cluster_alpha", "0.6"
        ])
    
    print(f"运行实验: {experiment_name}")
    print(f"数据集: {dataset}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print(f"实验 {experiment_name} 成功完成")
        else:
            print(f"实验 {experiment_name} 失败:")
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"实验 {experiment_name} 超时")
    except Exception as e:
        print(f"运行实验 {experiment_name} 时出错: {str(e)}")

def run_fashion_mnist_experiments():
    """运行Fashion-MNIST聚类对比实验"""
    print("=" * 60)
    print("开始Fashion-MNIST聚类对比实验...")
    print("=" * 60)
    
    # 实验1: 传统聚合方法
    run_experiment(
        experiment_name="FashionMNIST_Traditional_Aggregation",
        dataset="fashion_mnist",
        enable_clustering=False,
        rounds=50
    )
    
    # 实验2: 分层聚合 - 3个组，余弦相似度聚类
    run_experiment(
        experiment_name="FashionMNIST_Hierarchical_3Clusters_Cosine",
        dataset="fashion_mnist",
        enable_clustering=True,
        num_clusters=3,
        clustering_method='cosine_similarity',
        rounds=50
    )
    
    # 实验3: 分层聚合 - 4个组，余弦相似度聚类
    run_experiment(
        experiment_name="FashionMNIST_Hierarchical_4Clusters_Cosine",
        dataset="fashion_mnist",
        enable_clustering=True,
        num_clusters=4,
        clustering_method='cosine_similarity',
        rounds=50
    )
    
    # 实验4: 分层聚合 - 3个组，简单分组方法
    run_experiment(
        experiment_name="FashionMNIST_Hierarchical_3Clusters_Simple",
        dataset="fashion_mnist",
        enable_clustering=True,
        num_clusters=3,
        clustering_method='simple_grouping',
        rounds=50
    )

def run_cifar10_experiments():
    """运行CIFAR-10聚类对比实验"""
    print("=" * 60)
    print("开始CIFAR-10聚类对比实验...")
    print("=" * 60)
    
    # 实验1: 传统聚合方法
    run_experiment(
        experiment_name="CIFAR10_Traditional_Aggregation",
        dataset="cifar10",
        enable_clustering=False,
        rounds=100  # CIFAR-10通常需要更多轮次
    )
    
    # 实验2: 分层聚合 - 3个组，余弦相似度聚类
    run_experiment(
        experiment_name="CIFAR10_Hierarchical_3Clusters_Cosine",
        dataset="cifar10",
        enable_clustering=True,
        num_clusters=3,
        clustering_method='cosine_similarity',
        rounds=100
    )
    
    # 实验3: 分层聚合 - 4个组，余弦相似度聚类
    run_experiment(
        experiment_name="CIFAR10_Hierarchical_4Clusters_Cosine",
        dataset="cifar10",
        enable_clustering=True,
        num_clusters=4,
        clustering_method='cosine_similarity',
        rounds=100
    )
    
    # 实验4: 分层聚合 - 5个组，余弦相似度聚类（CIFAR-10复杂度更高，可以尝试更多组）
    run_experiment(
        experiment_name="CIFAR10_Hierarchical_5Clusters_Cosine",
        dataset="cifar10",
        enable_clustering=True,
        num_clusters=5,
        clustering_method='cosine_similarity',
        rounds=100
    )
    
    # 实验5: 分层聚合 - 3个组，简单分组方法
    run_experiment(
        experiment_name="CIFAR10_Hierarchical_3Clusters_Simple",
        dataset="cifar10",
        enable_clustering=True,
        num_clusters=3,
        clustering_method='simple_grouping',
        rounds=100
    )

def run_ablation_experiments():
    """运行消融实验，测试不同的cluster_alpha值"""
    print("=" * 60)
    print("开始消融实验（测试不同的cluster_alpha值）...")
    print("=" * 60)
    
    for alpha in [0.3, 0.5, 0.7]:
        # Fashion-MNIST消融实验
        cmd = [
            sys.executable, "main.py",
            "--running_name", f"FashionMNIST_Ablation_Alpha_{alpha}",
            "--dataset", "fashion_mnist",
            "--client_number", "10",
            "--rounds", "50",
            "--batch_size", "256",
            "--lr", "0.005",
            "--partition_method", "hetero",
            "--partition_alpha", "0.5",
            "--enable_clustering",
            "--num_clusters", "3",
            "--clustering_method", "cosine_similarity",
            "--cluster_alpha", str(alpha)
        ]
        
        print(f"运行Fashion-MNIST消融实验: cluster_alpha={alpha}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0:
                print(f"消融实验 alpha={alpha} 成功完成")
            else:
                print(f"消融实验 alpha={alpha} 失败")
        except Exception as e:
            print(f"消融实验 alpha={alpha} 出错: {str(e)}")

def main():
    """运行完整的对比实验"""
    
    print("开始完整的聚类对比实验...")
    print("包括Fashion-MNIST和CIFAR-10数据集")
    
    # 询问用户要运行哪些实验
    print("\n请选择要运行的实验:")
    print("1. Fashion-MNIST实验")
    print("2. CIFAR-10实验")
    print("3. 消融实验")
    print("4. 全部实验")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        run_fashion_mnist_experiments()
    elif choice == "2":
        run_cifar10_experiments()
    elif choice == "3":
        run_ablation_experiments()
    elif choice == "4":
        run_fashion_mnist_experiments()
        print("\n" + "="*60)
        run_cifar10_experiments()
        print("\n" + "="*60)
        run_ablation_experiments()
    else:
        print("无效选择，运行所有实验...")
        run_fashion_mnist_experiments()
        run_cifar10_experiments()
        run_ablation_experiments()
    
    print("\n" + "="*60)
    print("所有选定的实验完成!")

if __name__ == "__main__":
    main()