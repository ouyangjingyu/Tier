#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTFL自动化实验运行脚本
自动运行不同数据集和异质性参数的实验
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DTFLExperimentRunner:
    """DTFL实验自动化运行器"""
    
    def __init__(self):
        # 实验配置
        self.datasets = ['cifar100','cifar10']
        self.alphas = [0.1, 0.3, 0.5, 0.8, 1.0]
        
        # 基础参数(可根据需要修改)
        self.base_config = {
            'model': 'resnet56',
            'warmup_epochs': 1,
            'enable_warmup': True,
            'rounds': 2,
            'client_number': 10,
            'batch_size': 128,
            'lr': 0.005,
            'lr_factor': 0.9,
            'lr_patience': 10,
            'optimizer': 'Adam',
            'wd': 5e-4,
        }
        
        # 主程序路径
        self.main_script = 'main_warmup.py'
        
        # 实验记录
        self.experiment_log = {
            'start_time': None,
            'end_time': None,
            'experiments': [],
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0
        }
        
        # 创建必要的目录
        self._setup_directories()
    
    def _setup_directories(self):
        """创建必要的目录"""
        directories = ['./logs', './results', './data']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"确保目录存在: {directory}")
    
    def _build_command(self, dataset, alpha):
        """构建实验命令"""
        cmd = [
            'python', self.main_script,
            '--model', str(self.base_config['model']),
            '--dataset', dataset,
            '--partition_alpha', str(alpha),
            '--warmup_epochs', str(self.base_config['warmup_epochs']),
            '--enable_warmup', str(self.base_config['enable_warmup']),
            '--rounds', str(self.base_config['rounds']),
            '--client_number', str(self.base_config['client_number']),
            '--batch_size', str(self.base_config['batch_size']),
            '--lr', str(self.base_config['lr']),
            '--lr_factor', str(self.base_config['lr_factor']),
            '--lr_patience', str(self.base_config['lr_patience']),
            '--optimizer', str(self.base_config['optimizer']),
            '--wd', str(self.base_config['wd']),
        ]
        return cmd
    
    def _get_log_filename(self, dataset, alpha):
        """生成日志文件名"""
        return f"DTFL_{self.base_config['model']}_{dataset}_alpha{alpha}.txt"
    
    def _check_if_completed(self, dataset, alpha):
        """检查实验是否已完成"""
        log_file = os.path.join('./logs', self._get_log_filename(dataset, alpha))
        if os.path.exists(log_file):
            # 检查日志文件中是否包含完成标记
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Training and Evaluation completed!' in content:
                        return True
            except Exception as e:
                logger.warning(f"检查日志文件时出错: {e}")
        return False
    
    def run_single_experiment(self, dataset, alpha, experiment_id):
        """运行单个实验"""
        logger.info("=" * 80)
        logger.info(f"实验 {experiment_id}/{self.experiment_log['total_experiments']}")
        logger.info(f"数据集: {dataset}, Alpha: {alpha}")
        logger.info("=" * 80)
        
        # 检查是否已完成
        if self._check_if_completed(dataset, alpha):
            logger.info(f"实验已完成，跳过: {dataset}, alpha={alpha}")
            self.experiment_log['completed_experiments'] += 1
            return True
        
        # 构建命令
        cmd = self._build_command(dataset, alpha)
        
        # 记录实验信息
        experiment_info = {
            'id': experiment_id,
            'dataset': dataset,
            'alpha': alpha,
            'command': ' '.join(cmd),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'running'
        }
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # 运行实验
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出日志
            logger.info("开始实时输出...")
            for line in process.stdout:
                print(line, end='')  # 实时打印到控制台
            
            # 等待进程完成
            return_code = process.wait()
            
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                logger.info(f"✓ 实验成功完成!")
                logger.info(f"耗时: {elapsed_time/60:.2f} 分钟")
                experiment_info['status'] = 'completed'
                experiment_info['elapsed_time'] = elapsed_time
                self.experiment_log['completed_experiments'] += 1
                success = True
            else:
                logger.error(f"✗ 实验失败! 返回码: {return_code}")
                experiment_info['status'] = 'failed'
                experiment_info['return_code'] = return_code
                self.experiment_log['failed_experiments'] += 1
                success = False
            
            # 获取stderr输出
            stderr_output = process.stderr.read()
            if stderr_output:
                logger.error(f"错误输出: {stderr_output}")
                experiment_info['error'] = stderr_output
            
        except Exception as e:
            logger.error(f"✗ 实验执行异常: {str(e)}")
            experiment_info['status'] = 'error'
            experiment_info['error'] = str(e)
            self.experiment_log['failed_experiments'] += 1
            success = False
        
        experiment_info['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.experiment_log['experiments'].append(experiment_info)
        
        return success
    
    def run_all_experiments(self, skip_completed=True):
        """运行所有实验"""
        logger.info("\n" + "=" * 80)
        logger.info("DTFL 自动化实验运行器")
        logger.info("=" * 80)
        
        # 统计总实验数
        total = len(self.datasets) * len(self.alphas)
        self.experiment_log['total_experiments'] = total
        self.experiment_log['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"总实验数: {total}")
        logger.info(f"数据集: {self.datasets}")
        logger.info(f"Alpha值: {self.alphas}")
        logger.info(f"基础配置: {json.dumps(self.base_config, indent=2)}")
        logger.info("=" * 80 + "\n")
        
        # 运行所有实验组合
        experiment_id = 0
        for dataset in self.datasets:
            for alpha in self.alphas:
                experiment_id += 1
                
                success = self.run_single_experiment(dataset, alpha, experiment_id)
                
                # 每个实验后保存进度
                self._save_progress()
                
                # 实验间隔(可选)
                if experiment_id < total:
                    logger.info("\n等待5秒后开始下一个实验...\n")
                    time.sleep(5)
        
        self.experiment_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_progress()
        self._print_summary()
    
    def _save_progress(self):
        """保存实验进度"""
        progress_file = './results/experiment_progress.json'
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)
            logger.info(f"进度已保存到: {progress_file}")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def _print_summary(self):
        """打印实验总结"""
        logger.info("\n" + "=" * 80)
        logger.info("实验总结")
        logger.info("=" * 80)
        logger.info(f"开始时间: {self.experiment_log['start_time']}")
        logger.info(f"结束时间: {self.experiment_log['end_time']}")
        logger.info(f"总实验数: {self.experiment_log['total_experiments']}")
        logger.info(f"完成数量: {self.experiment_log['completed_experiments']}")
        logger.info(f"失败数量: {self.experiment_log['failed_experiments']}")
        
        # 计算总耗时
        if self.experiment_log['start_time'] and self.experiment_log['end_time']:
            start = datetime.strptime(self.experiment_log['start_time'], '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(self.experiment_log['end_time'], '%Y-%m-%d %H:%M:%S')
            total_time = (end - start).total_seconds()
            logger.info(f"总耗时: {total_time/3600:.2f} 小时")
        
        logger.info("\n详细结果:")
        for exp in self.experiment_log['experiments']:
            status_icon = "✓" if exp['status'] == 'completed' else "✗"
            logger.info(f"{status_icon} [{exp['id']}] {exp['dataset']}, alpha={exp['alpha']} - {exp['status']}")
        
        logger.info("=" * 80)
        
        # 生成报告
        self._generate_report()
    
    def _generate_report(self):
        """生成实验报告"""
        report_file = './results/experiment_report.txt'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DTFL 自动化实验报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"开始时间: {self.experiment_log['start_time']}\n")
                f.write(f"结束时间: {self.experiment_log['end_time']}\n\n")
                
                f.write("实验配置:\n")
                f.write("-" * 80 + "\n")
                for key, value in self.base_config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("实验统计:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  总实验数: {self.experiment_log['total_experiments']}\n")
                f.write(f"  完成数量: {self.experiment_log['completed_experiments']}\n")
                f.write(f"  失败数量: {self.experiment_log['failed_experiments']}\n\n")
                
                f.write("实验详情:\n")
                f.write("-" * 80 + "\n")
                for exp in self.experiment_log['experiments']:
                    f.write(f"\n实验 {exp['id']}:\n")
                    f.write(f"  数据集: {exp['dataset']}\n")
                    f.write(f"  Alpha: {exp['alpha']}\n")
                    f.write(f"  状态: {exp['status']}\n")
                    f.write(f"  开始时间: {exp['start_time']}\n")
                    f.write(f"  结束时间: {exp.get('end_time', 'N/A')}\n")
                    if 'elapsed_time' in exp:
                        f.write(f"  耗时: {exp['elapsed_time']/60:.2f} 分钟\n")
                    if 'error' in exp:
                        f.write(f"  错误: {exp['error']}\n")
                    f.write(f"  日志文件: ./logs/{self._get_log_filename(exp['dataset'], exp['alpha'])}\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            logger.info(f"实验报告已生成: {report_file}")
        except Exception as e:
            logger.error(f"生成报告失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DTFL自动化实验运行器')
    parser.add_argument('--skip-completed', action='store_true', default=True,
                       help='跳过已完成的实验 (默认: True)')
    parser.add_argument('--datasets', nargs='+', default=['cifar100','cifar10'],
                       help='要运行的数据集列表')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.8, 1.0],
                       help='要测试的alpha值列表')
    parser.add_argument('--rounds', type=int, default=2,
                       help='训练轮次')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                       help='预热轮次')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = DTFLExperimentRunner()
    
    # 更新配置
    if args.datasets:
        runner.datasets = args.datasets
    if args.alphas:
        runner.alphas = args.alphas
    if args.rounds:
        runner.base_config['rounds'] = args.rounds
    if args.warmup_epochs is not None:
        runner.base_config['warmup_epochs'] = args.warmup_epochs
    
    # 检查主程序是否存在
    if not os.path.exists(runner.main_script):
        logger.error(f"找不到主程序: {runner.main_script}")
        logger.error("请确保 main_complete_warmup.py 在当前目录下")
        sys.exit(1)
    
    # 运行所有实验
    try:
        runner.run_all_experiments(skip_completed=args.skip_completed)
    except KeyboardInterrupt:
        logger.warning("\n\n实验被用户中断!")
        runner._save_progress()
        runner._print_summary()
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        runner._save_progress()


if __name__ == '__main__':
    main()