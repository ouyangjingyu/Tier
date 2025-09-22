#!/usr/bin/env python3
"""
快速测试脚本 - 验证三种聚合方法的基本功能
运行时间较短，用于代码调试和功能验证
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuickTest")

def run_quick_test(dataset='fashion_mnist', aggregation_strategy='traditional', rounds=5):
    """运行快速测试"""
    
    logger.info(f"开始快速测试 - 数据集: {dataset}, 聚合策略: {aggregation_strategy}")
    
    # 构建命令
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--aggregation_strategy', aggregation_strategy,
        '--client_number', '3',  # 减少客户端数量
        '--rounds', str(rounds),  # 减少训练轮数
        '--partition_alpha', '0.5',
        '--batch_size', '128',  # 减少批次大小
        '--num_clusters', '2',
        '--running_name', f'quick_test_{aggregation_strategy}',
        '--enable_clustering'
    ]
    
    # 设置环境变量解决MKL兼容性问题
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    env['MKL_THREADING_LAYER'] = 'GNU'
    env['OMP_NUM_THREADS'] = '1'  # 减少线程冲突
    env['CUDA_VISIBLE_DEVICES'] = '0'  # 限制只使用一个GPU
    
    start_time = time.time()
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)  # 10分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"测试成功完成 - 耗时: {duration:.2f}秒")
            
            # 从输出中提取关键信息
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # 查看最后20行
                if '最佳准确率' in line or '训练完成' in line:
                    logger.info(f"结果: {line.strip()}")
            
            return True, duration
        else:
            logger.error(f"测试失败:")
            logger.error(f"stderr: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        logger.error("测试超时")
        return False, time.time() - start_time
    except Exception as e:
        logger.error(f"测试出现异常: {str(e)}")
        return False, time.time() - start_time

def main():
    """主函数 - 测试三种聚合策略"""
    
    print("快速测试脚本 - 验证聚合方法功能")
    print("=" * 50)
    
    # 测试配置
    test_configs = [
        ('fashion_mnist', 'traditional'),
        ('fashion_mnist', 'hierarchical'), 
        ('fashion_mnist', 'knowledge_distillation'),
        ('cifar10', 'traditional'),
        ('cifar10', 'knowledge_distillation')
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, (dataset, strategy) in enumerate(test_configs, 1):
        print(f"\n测试 {i}/{len(test_configs)}: {dataset} + {strategy}")
        print("-" * 30)
        
        success, duration = run_quick_test(dataset, strategy, rounds=3)
        
        results.append({
            'dataset': dataset,
            'strategy': strategy,
            'success': success,
            'duration': duration
        })
        
        if success:
            print(f"✅ 成功 - 耗时: {duration:.1f}秒")
        else:
            print(f"❌ 失败 - 耗时: {duration:.1f}秒")
        
        # 测试间隔
        time.sleep(2)
    
    # 总结结果
    total_time = time.time() - total_start_time
    successful_tests = sum(1 for r in results if r['success'])
    
    print(f"\n" + "=" * 50)
    print("测试总结:")
    print(f"- 总测试数: {len(test_configs)}")
    print(f"- 成功数: {successful_tests}")
    print(f"- 失败数: {len(test_configs) - successful_tests}")
    print(f"- 总耗时: {total_time:.1f}秒")
    
    print("\n详细结果:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['dataset']} + {result['strategy']}: {result['duration']:.1f}秒")
    
    if successful_tests == len(test_configs):
        print("\n🎉 所有测试通过！代码功能正常。")
        return 0
    else:
        print(f"\n⚠️  有 {len(test_configs) - successful_tests} 个测试失败，请检查代码。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)