#!/usr/bin/env python3
"""
聚合方法对比测试脚本
对比传统聚合、分层聚合和知识蒸馏聚合三种方法的性能
"""

import os
import sys
import subprocess
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"aggregation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("AggregationComparison")

class AggregationComparisonTester:
    """聚合方法对比测试器"""
    
    def __init__(self, output_dir="./comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试配置
        self.test_configs = {
            'datasets': ['cifar10', 'fashion_mnist'],
            'aggregation_strategies': ['traditional', 'hierarchical', 'knowledge_distillation'],
            'client_numbers': [5, 10],
            'rounds': 50,  # 为了快速测试，减少轮数
            'partition_alpha': 0.5,  # 固定50%异质性
            'batch_size': 256,
            'num_clusters': 3
        }
        
        # 结果存储
        self.results = []
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def run_single_experiment(self, dataset, aggregation_strategy, client_number, experiment_name):
        """运行单个实验"""
        logger.info(f"开始实验: {experiment_name}")
        
        # 构建命令行参数
        cmd = [
            'python', 'main.py',
            '--dataset', dataset,
            '--aggregation_strategy', aggregation_strategy,
            '--client_number', str(client_number),
            '--rounds', str(self.test_configs['rounds']),
            '--partition_alpha', str(self.test_configs['partition_alpha']),
            '--batch_size', str(self.test_configs['batch_size']),
            '--num_clusters', str(self.test_configs['num_clusters']),
            '--running_name', experiment_name,
            '--enable_clustering'  # 启用聚类
        ]
        
        # 设置环境变量解决MKL兼容性问题
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'GNU'
        env['OMP_NUM_THREADS'] = '1'  # 减少线程冲突
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行实验
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)  # 1小时超时
            
            # 记录结束时间
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"实验 {experiment_name} 成功完成，耗时: {duration:.2f}秒")
                
                # 解析结果
                experiment_result = self._parse_experiment_result(
                    experiment_name, dataset, aggregation_strategy, 
                    client_number, duration, result.stdout
                )
                
                self.results.append(experiment_result)
                return True
            else:
                logger.error(f"实验 {experiment_name} 失败:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"实验 {experiment_name} 超时")
            return False
        except Exception as e:
            logger.error(f"实验 {experiment_name} 出现异常: {str(e)}")
            return False
    
    def _parse_experiment_result(self, experiment_name, dataset, aggregation_strategy, 
                                client_number, duration, stdout):
        """解析实验结果"""
        result = {
            'experiment_name': experiment_name,
            'dataset': dataset,
            'aggregation_strategy': aggregation_strategy,
            'client_number': client_number,
            'duration_seconds': duration,
            'partition_alpha': self.test_configs['partition_alpha'],
            'rounds': self.test_configs['rounds'],
            'timestamp': datetime.now().isoformat()
        }
        
        # 从stdout中提取关键指标
        lines = stdout.split('\n')
        
        # 寻找最佳准确率
        best_accuracy = 0.0
        final_accuracy = 0.0
        
        for line in lines:
            if '最佳准确率:' in line:
                try:
                    # 提取数字
                    parts = line.split('最佳准确率:')
                    if len(parts) > 1:
                        acc_str = parts[1].split('%')[0].strip()
                        best_accuracy = float(acc_str)
                except:
                    pass
            elif '全局模型在独立测试集上的准确率:' in line:
                try:
                    # 提取最后一轮的准确率
                    parts = line.split('准确率:')
                    if len(parts) > 1:
                        acc_str = parts[1].split('%')[0].strip()
                        final_accuracy = float(acc_str)
                except:
                    pass
        
        result['best_accuracy'] = best_accuracy
        result['final_accuracy'] = final_accuracy
        
        logger.info(f"解析结果 - 最佳准确率: {best_accuracy:.2f}%, 最终准确率: {final_accuracy:.2f}%")
        
        return result
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("开始运行聚合方法对比实验")
        
        total_experiments = (len(self.test_configs['datasets']) * 
                           len(self.test_configs['aggregation_strategies']) * 
                           len(self.test_configs['client_numbers']))
        
        logger.info(f"总计需要运行 {total_experiments} 个实验")
        
        experiment_count = 0
        
        for dataset in self.test_configs['datasets']:
            for aggregation_strategy in self.test_configs['aggregation_strategies']:
                for client_number in self.test_configs['client_numbers']:
                    experiment_count += 1
                    
                    experiment_name = f"{self.experiment_id}_{dataset}_{aggregation_strategy}_c{client_number}"
                    
                    logger.info(f"运行实验 {experiment_count}/{total_experiments}: {experiment_name}")
                    
                    success = self.run_single_experiment(
                        dataset, aggregation_strategy, client_number, experiment_name
                    )
                    
                    if not success:
                        logger.warning(f"实验 {experiment_name} 失败，继续下一个实验")
                    
                    # 在实验之间稍作停顿
                    time.sleep(5)
        
        logger.info("所有实验完成")
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_comparison_report()
    
    def _save_results(self):
        """保存实验结果"""
        # 保存为JSON
        json_file = self.output_dir / f"results_{self.experiment_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"results_{self.experiment_id}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"结果已保存至: {json_file} 和 {csv_file}")
    
    def _generate_comparison_report(self):
        """生成对比报告"""
        if not self.results:
            logger.warning("没有实验结果，无法生成报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 生成摘要报告
        self._generate_summary_report(df)
        
        # 生成可视化图表
        self._generate_visualization(df)
        
        # 生成详细分析
        self._generate_detailed_analysis(df)
    
    def _generate_summary_report(self, df):
        """生成摘要报告"""
        logger.info("生成摘要报告...")
        
        report_file = self.output_dir / f"summary_report_{self.experiment_id}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 聚合方法对比实验报告\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"实验ID: {self.experiment_id}\n\n")
            
            # 实验配置
            f.write("## 实验配置\n\n")
            f.write(f"- 数据集: {', '.join(self.test_configs['datasets'])}\n")
            f.write(f"- 聚合策略: {', '.join(self.test_configs['aggregation_strategies'])}\n")
            f.write(f"- 客户端数量: {', '.join(map(str, self.test_configs['client_numbers']))}\n")
            f.write(f"- 训练轮数: {self.test_configs['rounds']}\n")
            f.write(f"- 数据异质性参数: {self.test_configs['partition_alpha']} (50%异质性)\n")
            f.write(f"- 聚类数量: {self.test_configs['num_clusters']}\n\n")
            
            # 总体结果
            f.write("## 总体结果\n\n")
            
            # 按聚合策略分组的平均准确率
            strategy_results = df.groupby('aggregation_strategy').agg({
                'best_accuracy': ['mean', 'std'],
                'final_accuracy': ['mean', 'std'],
                'duration_seconds': ['mean', 'std']
            }).round(3)
            
            f.write("### 按聚合策略分组的结果\n\n")
            f.write("| 聚合策略 | 最佳准确率(%) | 最终准确率(%) | 平均耗时(秒) |\n")
            f.write("|----------|---------------|---------------|-------------|\n")
            
            for strategy in self.test_configs['aggregation_strategies']:
                if strategy in strategy_results.index:
                    best_acc_mean = strategy_results.loc[strategy, ('best_accuracy', 'mean')]
                    best_acc_std = strategy_results.loc[strategy, ('best_accuracy', 'std')]
                    final_acc_mean = strategy_results.loc[strategy, ('final_accuracy', 'mean')]
                    final_acc_std = strategy_results.loc[strategy, ('final_accuracy', 'std')]
                    duration_mean = strategy_results.loc[strategy, ('duration_seconds', 'mean')]
                    
                    f.write(f"| {strategy} | {best_acc_mean:.2f}±{best_acc_std:.2f} | "
                           f"{final_acc_mean:.2f}±{final_acc_std:.2f} | {duration_mean:.0f} |\n")
            
            f.write("\n")
            
            # 按数据集分组的结果
            dataset_results = df.groupby(['dataset', 'aggregation_strategy']).agg({
                'best_accuracy': 'mean',
                'final_accuracy': 'mean'
            }).round(2)
            
            f.write("### 按数据集分组的结果\n\n")
            for dataset in self.test_configs['datasets']:
                f.write(f"#### {dataset.upper()}\n\n")
                f.write("| 聚合策略 | 最佳准确率(%) | 最终准确率(%) |\n")
                f.write("|----------|---------------|---------------|\n")
                
                for strategy in self.test_configs['aggregation_strategies']:
                    if (dataset, strategy) in dataset_results.index:
                        best_acc = dataset_results.loc[(dataset, strategy), 'best_accuracy']
                        final_acc = dataset_results.loc[(dataset, strategy), 'final_accuracy']
                        f.write(f"| {strategy} | {best_acc:.2f} | {final_acc:.2f} |\n")
                
                f.write("\n")
            
            # 结论
            f.write("## 结论\n\n")
            
            # 找出最佳策略
            best_strategy_overall = df.loc[df['best_accuracy'].idxmax(), 'aggregation_strategy']
            best_accuracy_overall = df['best_accuracy'].max()
            
            f.write(f"1. **最佳聚合策略**: {best_strategy_overall} (最高准确率: {best_accuracy_overall:.2f}%)\n")
            
            # 平均性能排名
            avg_performance = df.groupby('aggregation_strategy')['best_accuracy'].mean().sort_values(ascending=False)
            f.write(f"2. **平均性能排名**:\n")
            for i, (strategy, accuracy) in enumerate(avg_performance.items(), 1):
                f.write(f"   {i}. {strategy}: {accuracy:.2f}%\n")
            
            # 效率分析
            avg_time = df.groupby('aggregation_strategy')['duration_seconds'].mean().sort_values()
            f.write(f"3. **训练效率排名** (平均耗时):\n")
            for i, (strategy, time_taken) in enumerate(avg_time.items(), 1):
                f.write(f"   {i}. {strategy}: {time_taken:.0f}秒\n")
        
        logger.info(f"摘要报告已保存至: {report_file}")
    
    def _generate_visualization(self, df):
        """生成可视化图表"""
        logger.info("生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'聚合方法对比实验结果 - {self.experiment_id}', fontsize=16, fontweight='bold')
        
        # 1. 不同聚合策略的准确率对比
        ax1 = axes[0, 0]
        strategy_accuracy = df.groupby('aggregation_strategy')['best_accuracy'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(strategy_accuracy.index, strategy_accuracy.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('不同聚合策略的平均最佳准确率', fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 不同数据集上的性能对比
        ax2 = axes[0, 1]
        dataset_pivot = df.pivot_table(values='best_accuracy', index='aggregation_strategy', columns='dataset', aggfunc='mean')
        dataset_pivot.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('不同数据集上的聚合策略性能', fontweight='bold')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend(title='数据集')
        ax2.grid(True, alpha=0.3)
        
        # 3. 客户端数量对性能的影响
        ax3 = axes[1, 0]
        client_pivot = df.pivot_table(values='best_accuracy', index='aggregation_strategy', columns='client_number', aggfunc='mean')
        client_pivot.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_title('客户端数量对聚合策略性能的影响', fontweight='bold')
        ax3.set_ylabel('准确率 (%)')
        ax3.legend(title='客户端数量')
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练时间对比
        ax4 = axes[1, 1]
        time_data = df.groupby('aggregation_strategy')['duration_seconds'].mean().sort_values()
        bars4 = ax4.bar(time_data.index, time_data.values/60, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])  # 转换为分钟
        ax4.set_title('不同聚合策略的平均训练时间', fontweight='bold')
        ax4.set_ylabel('时间 (分钟)')
        
        # 添加数值标签
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}分', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / f"comparison_charts_{self.experiment_id}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图表已保存至: {chart_file}")
    
    def _generate_detailed_analysis(self, df):
        """生成详细分析"""
        logger.info("生成详细分析...")
        
        analysis_file = self.output_dir / f"detailed_analysis_{self.experiment_id}.md"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"# 详细分析报告\n\n")
            
            # 1. 聚合策略效果分析
            f.write("## 1. 聚合策略效果分析\n\n")
            
            for strategy in self.test_configs['aggregation_strategies']:
                strategy_data = df[df['aggregation_strategy'] == strategy]
                if not strategy_data.empty:
                    avg_acc = strategy_data['best_accuracy'].mean()
                    std_acc = strategy_data['best_accuracy'].std()
                    max_acc = strategy_data['best_accuracy'].max()
                    min_acc = strategy_data['best_accuracy'].min()
                    
                    f.write(f"### {strategy}\n")
                    f.write(f"- 平均准确率: {avg_acc:.2f}% (±{std_acc:.2f}%)\n")
                    f.write(f"- 最高准确率: {max_acc:.2f}%\n")
                    f.write(f"- 最低准确率: {min_acc:.2f}%\n")
                    f.write(f"- 稳定性评分: {100 - (std_acc/avg_acc*100):.1f}/100\n\n")
            
            # 2. 数据集适应性分析
            f.write("## 2. 数据集适应性分析\n\n")
            
            for dataset in self.test_configs['datasets']:
                f.write(f"### {dataset.upper()}\n")
                dataset_data = df[df['dataset'] == dataset]
                
                if not dataset_data.empty:
                    best_strategy_for_dataset = dataset_data.loc[dataset_data['best_accuracy'].idxmax(), 'aggregation_strategy']
                    best_acc_for_dataset = dataset_data['best_accuracy'].max()
                    
                    f.write(f"- 最佳聚合策略: {best_strategy_for_dataset} ({best_acc_for_dataset:.2f}%)\n")
                    
                    strategy_ranking = dataset_data.groupby('aggregation_strategy')['best_accuracy'].mean().sort_values(ascending=False)
                    f.write(f"- 策略排名:\n")
                    for i, (strategy, acc) in enumerate(strategy_ranking.items(), 1):
                        f.write(f"  {i}. {strategy}: {acc:.2f}%\n")
                f.write("\n")
            
            # 3. 扩展性分析
            f.write("## 3. 扩展性分析\n\n")
            
            for client_num in self.test_configs['client_numbers']:
                f.write(f"### {client_num}个客户端\n")
                client_data = df[df['client_number'] == client_num]
                
                if not client_data.empty:
                    best_strategy_for_clients = client_data.loc[client_data['best_accuracy'].idxmax(), 'aggregation_strategy']
                    best_acc_for_clients = client_data['best_accuracy'].max()
                    
                    f.write(f"- 最佳聚合策略: {best_strategy_for_clients} ({best_acc_for_clients:.2f}%)\n")
                    
                    avg_time = client_data.groupby('aggregation_strategy')['duration_seconds'].mean()
                    f.write(f"- 平均训练时间:\n")
                    for strategy, time_taken in avg_time.items():
                        f.write(f"  - {strategy}: {time_taken:.0f}秒\n")
                f.write("\n")
        
        logger.info(f"详细分析已保存至: {analysis_file}")

def main():
    """主函数"""
    print("聚合方法对比测试脚本")
    print("=" * 50)
    
    # 创建测试器
    tester = AggregationComparisonTester()
    
    # 运行所有实验
    tester.run_all_experiments()
    
    print("\n" + "=" * 50)
    print("所有实验完成！请查看结果文件夹中的报告。")

if __name__ == "__main__":
    main()