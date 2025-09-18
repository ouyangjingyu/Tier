#!/usr/bin/env python3
"""
CIFAR-10聚类功能快速测试脚本
用于验证CIFAR-10聚类功能是否正常工作
"""

import subprocess
import sys
import os

def run_quick_test():
    """运行快速测试"""
    
    print("开始CIFAR-10聚类功能快速测试...")
    print("使用较少的轮次进行快速验证")
    
    # 测试1: 传统聚合方法 (基线)
    print("\n" + "="*50)
    print("测试1: CIFAR-10传统聚合方法")
    print("="*50)
    
    cmd1 = [
        sys.executable, "main.py",
        "--running_name", "CIFAR10_Traditional_QuickTest",
        "--dataset", "cifar10",
        "--client_number", "8",
        "--rounds", "20",  # 快速测试，只训练20轮
        "--batch_size", "128",
        "--lr", "0.01",
        "--partition_method", "hetero",
        "--partition_alpha", "0.5"
    ]
    
    try:
        result = subprocess.run(cmd1, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        if result.returncode == 0:
            print("✓ 传统聚合方法测试成功")
        else:
            print("✗ 传统聚合方法测试失败")
            print("错误信息:", result.stderr[:500])
    except Exception as e:
        print(f"✗ 传统聚合方法测试出错: {str(e)}")
    
    # 测试2: 分层聚合方法
    print("\n" + "="*50)
    print("测试2: CIFAR-10分层聚合方法")
    print("="*50)
    
    cmd2 = [
        sys.executable, "main.py",
        "--running_name", "CIFAR10_Hierarchical_QuickTest",
        "--dataset", "cifar10",
        "--client_number", "8",
        "--rounds", "20",
        "--batch_size", "128",
        "--lr", "0.01",
        "--partition_method", "hetero",
        "--partition_alpha", "0.5",
        "--enable_clustering",
        "--num_clusters", "3",
        "--clustering_method", "cosine_similarity",
        "--cluster_alpha", "0.6"
    ]
    
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print("✓ 分层聚合方法测试成功")
        else:
            print("✗ 分层聚合方法测试失败")
            print("错误信息:", result.stderr[:500])
    except Exception as e:
        print(f"✗ 分层聚合方法测试出错: {str(e)}")
    
    print("\n" + "="*50)
    print("CIFAR-10聚类功能快速测试完成!")
    print("如果两个测试都成功，说明CIFAR-10聚类功能工作正常")
    print("="*50)

def run_clustering_verification():
    """运行聚类验证测试"""
    
    print("\n开始聚类算法验证...")
    
    # 测试不同聚类方法
    methods = ['cosine_similarity', 'simple_grouping']
    
    for method in methods:
        print(f"\n测试聚类方法: {method}")
        
        cmd = [
            sys.executable, "main.py",
            "--running_name", f"CIFAR10_Clustering_Verify_{method}",
            "--dataset", "cifar10",
            "--client_number", "6",  # 较少客户端，快速测试
            "--rounds", "10",       # 更少轮次
            "--batch_size", "64",
            "--lr", "0.01",
            "--partition_method", "hetero",
            "--partition_alpha", "0.3",  # 更强的异质性
            "--enable_clustering",
            "--num_clusters", "2",
            "--clustering_method", method,
            "--cluster_alpha", "0.5"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20分钟
            if result.returncode == 0:
                print(f"✓ 聚类方法 {method} 测试成功")
            else:
                print(f"✗ 聚类方法 {method} 测试失败")
                print("错误信息:", result.stderr[:300])
        except Exception as e:
            print(f"✗ 聚类方法 {method} 测试出错: {str(e)}")

def main():
    """主函数"""
    
    print("CIFAR-10聚类功能测试工具")
    print("=" * 60)
    
    choice = input("选择测试类型:\n1. 快速功能测试\n2. 聚类算法验证\n3. 全部测试\n请输入 (1-3): ").strip()
    
    if choice == "1":
        run_quick_test()
    elif choice == "2":
        run_clustering_verification()
    elif choice == "3":
        run_quick_test()
        run_clustering_verification()
    else:
        print("默认运行快速功能测试...")
        run_quick_test()

if __name__ == "__main__":
    main()