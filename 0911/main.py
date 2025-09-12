import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import random
import argparse
import logging
import wandb
import copy
import warnings
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import math

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入自定义模块
from model.resnet import EnhancedServerModel, TierAwareClientModel, ImprovedGlobalClassifier
from utils.simplified_aggregator import SimplifiedGlobalAggregator
from utils.simplified_trainer import SimplifiedSerialTrainer
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_loss import EnhancedStagedLoss

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='简化版PSFL: 个性化拆分联邦学习框架')
    
    # 实验标识
    parser.add_argument('--running_name', default="PSFL_Simplified", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.005, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.9, type=float, help='学习率衰减因子')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', help='使用的神经网络 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='fashion_mnist', 
                       help='训练数据集 (cifar10, cifar100, fashion_mnist, cinic10)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=5, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=5, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=256, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    
    # PSFL特有参数
    parser.add_argument('--init_alpha', default=0.6, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.15, type=float, help='初始特征对齐损失权重')
    
    args = parser.parse_args()
    return args

def setup_logging(args):
    # 配置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.running_name}.log")
        ]
    )
    logger = logging.getLogger("PSFL")
    
    # 初始化wandb
    try:
        wandb.init(
            mode="offline",
            project="PSFL_Simplified",
            name=args.running_name,
            config=args,
            tags=[f"model_{args.model}", f"dataset_{args.dataset}", 
                  f"clients_{args.client_number}", f"partition_{args.partition_method}"],
            group=f"{args.model}_{args.dataset}"
        )
        
        # 设置自定义面板
        wandb.define_metric("round")
        wandb.define_metric("global/*", step_metric="round")
        wandb.define_metric("local/*", step_metric="round")
        wandb.define_metric("split/*", step_metric="round")
        wandb.define_metric("time/*", step_metric="round")
        
    except Exception as e:
        print(f"警告: wandb初始化失败: {e}")
        
    return logger

def load_dataset(args):
    if args.dataset == "cifar10":
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    elif args.dataset == "fashion_mnist":
        data_loader = load_partition_data_fashion_mnist
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    else:
        data_loader = load_partition_data_cifar10

    if args.dataset == "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
        
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset

def allocate_device_resources(client_number):
    resources = {}
    
    # 随机分配tier (1-4)
    tier_weights = [0.2, 0.3, 0.3, 0.2]  # tier 1-4的分布概率
    tiers = random.choices(range(1, 5), weights=tier_weights, k=client_number)
    
    # 为每个客户端分配资源
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier == 1:  # 高性能设备
            compute_power = random.uniform(0.8, 1.0)
            network_speed = random.choice([50, 100, 200])
            storage_capacity = random.choice([256, 512, 1024])
        elif tier == 2:  # 中高性能设备
            compute_power = random.uniform(0.6, 0.8)
            network_speed = random.choice([30, 50, 100])
            storage_capacity = random.choice([128, 256, 512])
        elif tier == 3:  # 中低性能设备
            compute_power = random.uniform(0.3, 0.6)
            network_speed = random.choice([20, 30, 50])
            storage_capacity = random.choice([64, 128, 256])
        else:  # tier 4, 低性能设备
            compute_power = random.uniform(0.1, 0.3)
            network_speed = random.choice([5, 10, 20])
            storage_capacity = random.choice([16, 32, 64])
        
        # 存储资源信息
        resources[client_id] = {
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources

def load_global_test_set(args):
    """创建全局IID测试集用于评估泛化性能"""
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                [0.2675, 0.2565, 0.2761])
        ])
        
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "fashion_mnist":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530])
        ])
        
        testset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cinic10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    else:
        # 默认返回CIFAR10
        return load_global_test_set_cifar10(args)

def evaluate_global_model(client_models, server_models, global_test_loader, device, client_number):
    """
    评估全局模型在全局测试集上的性能
    在聚合后，所有客户端共享层和服务器模型都是统一的，随机选择一个客户端进行测试
    """
    # 随机选择一个客户端进行测试
    selected_client_id = random.randint(0, client_number - 1)
    
    # 获取选中客户端的模型
    client_model = client_models[selected_client_id].to(device)
    server_model = server_models[selected_client_id].to(device)
    
    # 设置为评估模式
    client_model.eval()
    server_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in global_test_loader:
            # 移到设备
            data, target = data.to(device), target.to(device)
            
            try:
                # 完整的前向传播：客户端共享层 -> 服务器模型 -> 分类结果
                _, shared_features, _ = client_model(data)
                logits = server_model(shared_features)
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            except Exception as e:
                logging.error(f"评估中出现错误: {str(e)}")
                continue
    
    accuracy = 100.0 * correct / max(1, total)
    
    logging.info(f"全局模型评估 - 使用客户端 {selected_client_id}, 样本总数: {total}, 正确预测: {correct}")
    
    return accuracy, selected_client_id

# 主函数
def main():
    """主函数，简化版PSFL实现"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 添加新参数
    args.initial_phase_rounds = 10  # 初始阶段轮数
    args.alternating_phase_rounds = 70  # 交替训练阶段轮数
    args.fine_tuning_phase_rounds = 20  # 精细调整阶段轮数
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化PSFL: 简化版本 - 每个客户端对应一个服务器模型")
    
    # 设置默认设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"默认设备: {device}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    
    # 获取数据集信息
    if args.dataset != "cinic10":
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 加载全局测试集
    logger.info("加载全局IID测试集用于评估泛化性能...")
    global_test_loader = load_global_test_set(args)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_device_resources(args.client_number)
    
    # 创建客户端管理器
    logger.info("创建客户端管理器...")
    client_manager = TierHFLClientManager()
    
    # 注册客户端
    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        tier = resource["tier"]
        
        # 创建客户端
        client = client_manager.add_client(
            client_id=client_id,
            tier=tier,
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            device=device,
            lr=args.lr,
            local_epochs=args.client_epoch
        )
        
        logger.info(f"客户端 {client_id} - Tier: {tier}, 训练样本数: {train_data_local_num_dict[client_id]}")

    # 确定输入通道数
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    # 创建客户端模型（每个客户端一个）
    logger.info(f"创建 {args.client_number} 个双路径客户端模型...")
    client_models = {}
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        client_models[client_id] = TierAwareClientModel(
            num_classes=class_num, 
            tier=tier,
            model_type=args.model,
            input_channels=input_channels
        )

    # 创建服务器模型（每个客户端对应一个，包含分类器）
    logger.info(f"创建 {args.client_number} 个服务器模型...")
    server_models = {}
    for client_id in range(args.client_number):
        server_models[client_id] = EnhancedServerModel(
            num_classes=class_num,
            model_type=args.model,
            input_channels=input_channels
        ).to(device)
    
    # 创建全局聚合器
    logger.info("创建全局聚合器...")
    global_aggregator = SimplifiedGlobalAggregator(
        client_data_sizes=train_data_local_num_dict,
        device=device
    )
    
    # 创建简化版串行训练器
    logger.info("创建简化版串行训练器...")
    trainer = SimplifiedSerialTrainer(
        client_manager=client_manager,
        client_models=client_models,
        server_models=server_models,
        device=device
    )
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 确定训练阶段
        if round_idx < args.initial_phase_rounds:
            training_phase = "initial"
            logger.info("当前处于初始特征学习阶段")
        elif round_idx < args.initial_phase_rounds + args.alternating_phase_rounds:
            training_phase = "alternating"
            logger.info("当前处于交替训练阶段")
        else:
            training_phase = "fine_tuning"
            logger.info("当前处于精细调整阶段")
        
        # 执行训练
        train_results, eval_results, shared_states, server_states, training_time = trainer.execute_round(
            round_idx=round_idx, 
            total_rounds=args.rounds,
            training_phase=training_phase
        )
        
        # 全局聚合
        logger.info("执行全局聚合...")
        aggregation_start_time = time.time()
        
        # 聚合共享层和服务器模型
        global_shared_layers, global_server_model = global_aggregator.aggregate(
            shared_states, server_states
        )
        
        # 更新所有客户端的共享层和服务器模型
        trainer.update_global_models(global_shared_layers, global_server_model)
        
        aggregation_time = time.time() - aggregation_start_time
        
        # 评估全局模型性能（在聚合后随机选择客户端测试）
        global_model_accuracy, selected_client_id = evaluate_global_model(
            client_models, server_models, global_test_loader, device, args.client_number
        )
        
        # 计算平均准确率
        avg_local_train_acc = np.mean([result.get('local_train_accuracy', 0) for result in eval_results.values()])
        avg_local_test_acc = np.mean([result.get('local_test_accuracy', 0) for result in eval_results.values()])
        avg_split_train_acc = np.mean([result.get('split_train_accuracy', 0) for result in eval_results.values()])
        avg_split_test_acc = np.mean([result.get('split_test_accuracy', 0) for result in eval_results.values()])
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            try:
                torch.save({
                    'client_models': {cid: model.state_dict() for cid, model in client_models.items()},
                    'server_models': {cid: model.state_dict() for cid, model in server_models.items()},
                    'global_shared_layers': global_shared_layers,
                    'global_server_model': global_server_model,
                    'round': round_idx,
                    'accuracy': best_accuracy
                }, f"{args.running_name}_best_model.pth")
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"本地模型 - 训练准确率: {avg_local_train_acc:.2f}%, 测试准确率: {avg_local_test_acc:.2f}%")
        logger.info(f"拆分模型 - 训练准确率: {avg_split_train_acc:.2f}%, 测试准确率: {avg_split_test_acc:.2f}%")
        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}% (使用客户端{selected_client_id})")
        logger.info(f"最佳准确率: {best_accuracy:.2f}%")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练: {training_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 记录到wandb
        try:
            metrics = {
                "round": round_idx + 1,
                "local/avg_train_accuracy": avg_local_train_acc,
                "local/avg_test_accuracy": avg_local_test_acc,
                "split/avg_train_accuracy": avg_split_train_acc,
                "split/avg_test_accuracy": avg_split_test_acc,
                "global/test_accuracy": global_model_accuracy,
                "global/best_accuracy": best_accuracy,
                "global/selected_client_id": selected_client_id,
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/aggregation_seconds": aggregation_time,
                "training/phase": 1 if training_phase == "initial" else (2 if training_phase == "alternating" else 3)
            }
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"记录wandb指标失败: {str(e)}")
        
        # 动态学习率调整
        if round_idx > 0 and round_idx % 10 == 0:
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    client.lr *= args.lr_factor
                    logger.info(f"客户端 {client_id} 学习率更新为: {client.lr:.6f}")
    
    # 训练完成
    logger.info(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")
    
    # 关闭wandb
    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()