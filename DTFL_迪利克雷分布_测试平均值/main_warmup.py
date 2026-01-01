# ============================================================================
# Deployment Environment and Resource Profiles:
# The DTFL and the baselines are deployed on a server with the following specifications:
# - Dual-sockets Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz 
# - Four NVIDIA GeForce GTX 1080 Ti GPUs
# - 64 GB of memory

# Modified Version with:
# 1. Warmup Phase (10 local epochs before federated training)
# 2. Log output to file (DTFL_model_dataset_alpha.txt)
# ============================================================================

import torch
from torch import nn
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np
import os

import time
import sys
import argparse
import logging
from datetime import datetime

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from model.resnet import resnet56_SFL_local_tier_7
from model.resnet import resnet110_SFL_fedavg_base
from model.resnet import resnet110_SFL_local_tier_7
from model.resnet import resnet56_SFL_fedavg_base

from utils.loss import PatchShuffle
from utils.loss import dis_corr
from utils.fedavg import aggregated_fedavg

from utils.TierScheduler import TierScheduler
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist

import matplotlib
matplotlib.use('Agg')
import copy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "Multi-Tier Splitfed Local Loss with Warmup"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

def add_args(parser):
    
    parser.add_argument('--running_name', default="DTFL_Warmup", type=str)
    
    # Optimization related arguments
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_patience', default=10, type=float)
    parser.add_argument('--lr_min', default=0, type=float)
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=1e-4)
 
    # Model related arguments
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')
    
    # Data loading and preprocessing related arguments
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='dataset used for training: cifar10, cifar100, cinic10, fashion_mnist')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
        
    # Federated learning related arguments
    parser.add_argument('--client_epoch', default=1, type=int)
    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--whether_local_loss', default=True, type=bool)
    parser.add_argument('--tier', default=5, type=int)
        
    # Privacy related arguments
    parser.add_argument('--whether_dcor', default=False, type=bool)
    parser.add_argument('--dcor_coefficient', default=0.5, type=float)
    parser.add_argument('--PatchShuffle', default=0, type=int)  
    
    # Warmup related arguments
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='number of local warmup epochs before federated training')
    parser.add_argument('--enable_warmup', default=True, type=bool,
                        help='whether to enable warmup phase')
    
    # Simulation arguments
    parser.add_argument('--net_speed_list', type=str, default=[100, 30, 30, 30, 10], 
                    metavar='N', help='list of net speeds in mega bytes')
    parser.add_argument('--delay_coefficient_list', type=str, default=[16, 20, 34, 130, 250],
                    metavar='N', help='list of delay coefficients')
    
    args = parser.parse_args()
    return args


# ============================================================================
# Setup Logging to File
# ============================================================================
def setup_logging(args):
    """Setup logging to both console and file"""
    # Create logs directory if not exists
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename: DTFL_model_dataset_alpha.txt
    log_filename = f"DTFL_{args.model}_{args.dataset}_alpha{args.partition_alpha}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    logger.info("="*80)
    logger.info(f"Experiment: DTFL with Warmup Phase")
    logger.info(f"Log file: {log_path}")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Partition Method: {args.partition_method}")
    logger.info(f"  Partition Alpha: {args.partition_alpha}")
    logger.info(f"  Warmup Enabled: {args.enable_warmup}")
    logger.info(f"  Warmup Epochs: {args.warmup_epochs if args.enable_warmup else 0}")
    logger.info(f"  Training Rounds: {args.rounds}")
    logger.info(f"  Client Number: {args.client_number}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  LR Factor: {args.lr_factor}")
    logger.info(f"  LR Patience: {args.lr_patience}")
    logger.info(f"  Optimizer: {args.optimizer}")
    logger.info(f"  Weight Decay: {args.wd}")
    logger.info("="*80)
    
    return logger


DYNAMIC_LR_THRESHOLD = 0.0001
DEFAULT_FRAC = 1.0

NUM_CPUs = os.cpu_count()

parser = argparse.ArgumentParser()
args = add_args(parser)

# Setup logging first
logger = setup_logging(args)
logger.info(f"Arguments: {args}")

# Model selection
SFL_local_tier = resnet56_SFL_local_tier_7

if args.dataset == 'cifar10' or args.dataset == 'fashion_mnist':
    class_num = 10
elif args.dataset == 'cifar100' or args.dataset == 'cinic10':
    class_num = 100

if args.model == 'resnet110':
    SFL_local_tier = resnet110_SFL_local_tier_7
    num_tiers = 7
    init_glob_model = resnet110_SFL_fedavg_base(classes=class_num, tier=1, fedavg_base=True)

if args.model == 'resnet56':
    SFL_local_tier = resnet56_SFL_local_tier_7
    num_tiers = 7
    init_glob_model = resnet56_SFL_fedavg_base(classes=class_num, tier=1, fedavg_base=True)

whether_local_loss = args.whether_local_loss
whether_dcor = args.whether_dcor
dcor_coefficient = args.dcor_coefficient
tier = args.tier
client_epoch = args.client_epoch
client_epoch = np.ones(args.client_number, dtype=int) * client_epoch

client_type_percent = [0.0, 0.0, 0.0, 0.0, 1.0]

if num_tiers == 7:
    client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    tier = 1

client_number_tier = (np.dot(args.client_number, client_type_percent))

# Network speed profile
net_speed_list = list(np.array(args.net_speed_list) * 1024 ** 2)
net_speed = net_speed_list * (args.client_number // 5 + 1)
delay_coefficient_list = list(np.array(args.delay_coefficient_list) / 14.5)
delay_coefficient = delay_coefficient_list * (args.client_number // 5 + 1)
delay_coefficient = list(np.array(delay_coefficient))

num_users = args.client_number
epochs = args.rounds
lr = args.lr

# Global variables for data transmission
global data_transmit
model_parameter_data_size = 0
intermediate_data_size = 0

# Load dataset
def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    elif dataset_name == "fashion_mnist":
        data_loader = load_partition_data_fashion_mnist
    else:
        data_loader = load_partition_data_cifar10

    if dataset_name == "cinic10":
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

logger.info("Loading dataset...")
if args.dataset != "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
if args.dataset == "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts] = dataset
    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    dataset_size = {}
    for i in range(0, len(traindata_cls_counts)):
        dataset_size[i] = sum(traindata_cls_counts[i].values())
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)

dataset_size = {}
if args.dataset != "cinic10":
    for i in range(0, args.client_number):
        dataset_size[i] = len(dataset_train[i].dataset.target)
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)

logger.info(f"Dataset loaded. Train samples: {train_data_num}, Test samples: {test_data_num}")


# Functions
def get_random_user_indices(num_users, DEFAULT_FRAC=0.1):
    m = max(int(DEFAULT_FRAC * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    return idxs_users, m


def calculate_data_size(w_model):
    """Calculate the data size (memory usage) of tensors in the model"""
    data_size = 0
    for k in w_model:
        data_size += sys.getsizeof(w_model[k].storage())
    return data_size


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


def compute_delay(data_transmitted_client: float, net_speed: float, delay_coefficient: float, duration) -> float:
    """Compute simulated delay based on network and computation time"""
    net_delay = data_transmitted_client / net_speed
    computation_delay = duration * delay_coefficient
    total_delay = net_delay + computation_delay
    simulated_delay = total_delay
    return simulated_delay


# ============================================================================
# Warmup Phase Implementation
# ============================================================================
def warmup_phase(args, net_glob_client_tier, client_tier, dataset_train, device, num_users):
    """
    Warmup phase: Each client trains locally for warmup_epochs without any aggregation
    
    Args:
        args: Command line arguments
        net_glob_client_tier: Dictionary of client models for each tier
        client_tier: Dictionary mapping client index to tier
        dataset_train: Training dataset for each client
        device: Computing device (CPU/GPU)
        num_users: Number of clients
    
    Returns:
        warmed_up_models: Dictionary of client models after warmup
    """
    if not args.enable_warmup:
        logger.info("Warmup phase disabled. Skipping...")
        return None
    
    logger.info("="*80)
    logger.info("Starting Warmup Phase")
    logger.info(f"Warmup Epochs: {args.warmup_epochs}")
    logger.info(f"Clients: {num_users}")
    logger.info("="*80)
    
    warmed_up_models = {}
    criterion = nn.CrossEntropyLoss()
    
    warmup_start_time = time.time()
    
    for client_idx in range(num_users):
        logger.info(f"\n{'='*60}")
        logger.info(f"Client {client_idx} Warmup Training")
        logger.info(f"{'='*60}")
        
        # Get client's tier and model
        client_model_tier = client_tier[client_idx]
        net_local = copy.deepcopy(net_glob_client_tier[client_model_tier]).to(device)
        net_local.train()
        
        # Setup optimizer for warmup
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net_local.parameters(), lr=args.lr, 
                                        weight_decay=args.wd, amsgrad=True)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net_local.parameters(), lr=args.lr, 
                                       momentum=0.9, nesterov=True, weight_decay=args.wd)
        
        # Get client's training data
        train_loader = dataset_train[client_idx]
        
        client_warmup_start = time.time()
        
        # Local warmup training
        for epoch in range(args.warmup_epochs):
            epoch_loss = []
            epoch_acc = []
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                extracted_features, fx = net_local(images)
                
                # Calculate loss (local loss)
                labels = labels.to(torch.long)
                loss = criterion(extracted_features, labels)
                
                # Add distance correlation if enabled
                if whether_dcor:
                    Dcor_value = dis_corr(images, fx)
                    loss = (1 - dcor_coefficient) * loss + dcor_coefficient * Dcor_value
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                acc = calculate_accuracy(extracted_features, labels)
                
                epoch_loss.append(loss.item())
                epoch_acc.append(acc.item())
            
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            avg_acc = sum(epoch_acc) / len(epoch_acc)
            
            # Log every 2 epochs or the last epoch
            if (epoch + 1) % 2 == 0 or epoch == args.warmup_epochs - 1:
                logger.info(f"  Epoch {epoch+1:2d}/{args.warmup_epochs}: "
                          f"Loss = {avg_loss:.4f}, Acc = {avg_acc:.2f}%")
        
        client_warmup_time = time.time() - client_warmup_start
        
        # Save warmed-up model
        warmed_up_models[client_idx] = net_local.state_dict()
        
        logger.info(f"Client {client_idx} warmup completed in {client_warmup_time:.2f}s")
        logger.info(f"Final: Loss = {avg_loss:.4f}, Acc = {avg_acc:.2f}%")
    
    total_warmup_time = time.time() - warmup_start_time
    
    logger.info("="*80)
    logger.info(f"Warmup Phase Completed in {total_warmup_time:.2f}s")
    logger.info("="*80)
    
    return warmed_up_models


# Initialize models
logger.info("Initializing models...")
net_glob_client_tier = {}
net_glob_client, _ = SFL_local_tier(classes=class_num, tier=tier)
for i in range(1, num_tiers+1):
    net_glob_client_tier[i], _ = SFL_local_tier(classes=class_num, tier=i)

if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
    net_glob_client = nn.DataParallel(net_glob_client, device_ids=list(range(torch.cuda.device_count())))
    for i in range(1, num_tiers+1):
        net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i], 
                                                   device_ids=list(range(torch.cuda.device_count())))

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)
net_glob_client.to(device)

# Initialize server models
net_glob_server_tier = {}
_, net_glob_server = SFL_local_tier(classes=class_num, tier=tier)
for i in range(1, num_tiers+1):
    _, net_glob_server_tier[i] = SFL_local_tier(classes=class_num, tier=i)

if torch.cuda.device_count() > 1:
    net_glob_server = nn.DataParallel(net_glob_server, device_ids=list(range(torch.cuda.device_count())))
    for i in range(1, num_tiers+1):
        net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i], 
                                                   device_ids=list(range(torch.cuda.device_count())))

for i in range(1, num_tiers+1):
    net_glob_server_tier[i].to(device)
net_glob_server.to(device)

# Server-side variables
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

time_train_server_train = 0
time_train_server_train_all = 0

# FedAvg functions
def FedAvg(w):
    len_min = float('inf')
    index_len_min = 0
    for j in range(0, len(w)):
        if len(w[j]) < len_min:
            len_min = len(w[j])
            index_len_min = j
    w[0], w[index_len_min] = w[index_len_min], w[0]
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        c = 1
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            c += 1
        w_avg[k] = torch.div(w_avg[k], c)
    return w_avg


# to print train - test together in each round
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
best_acc = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_glob_server_tier = {}
net_glob_server_tier[tier].load_state_dict(w_glob_server)
for i in range(1, num_tiers+1):
    w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
w_locals_server = []
w_locals_server_tier = {}
for i in range(1, num_tiers+1):
    w_locals_server_tier[i] = []

# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False

# Initialization of net_model_server and net_server
net_model_server_tier = {}
net_model_client_tier = {}
client_tier = {}
for i in range(0, num_users):
    client_tier[i] = num_tiers

k = 0
net_model_server = [net_glob_server for i in range(num_users)]
for i in range(len(client_number_tier)):
    for j in range(int(client_number_tier[i])):
        net_model_server_tier[k] = net_glob_server_tier[i+1]
        client_tier[k] = i+1
        k += 1

net_server = copy.deepcopy(net_model_server[0]).to(device)
net_server = copy.deepcopy(net_model_server_tier[0]).to(device)

optimizer_server_glob = torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True)
scheduler_server = ReduceLROnPlateau(optimizer_server_glob, 'max', factor=0.8, patience=0, threshold=0.0000001)
patience = args.lr_patience
factor = args.lr_factor
wait = 0
new_lr = lr
min_lr = args.lr_min

times_in_server = []


# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, extracted_features):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr
    time_train_server_s = time.time()
    
    net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)
    
    net_server.train()
    lr = new_lr
    if args.optimizer == "Adam":
        optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True)
    elif args.optimizer == "SGD":
        optimizer_server = torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
                                          nesterov=True,
                                          weight_decay=args.wd)
    
    time_train_server_s = time.time()
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    # forward prop
    fx_server = net_server(fx_client)
    
    # calculate loss
    y = y.to(torch.long)
    loss = criterion(fx_server, y)
    
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    # backward prop
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)
    net_model_server_tier[idx] = copy.deepcopy(net_server)
    time_train_server_train += time.time() - time_train_server_s
    
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        times_in_server.append(time_train_server_train)
        time_train_server_train_all += time_train_server_train
        total_time += time_train_server_train
        time_train_server_train = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.2f} \tLoss: {:.3f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch
        w_server = net_server.state_dict()
        
        # If one local epoch is completed
        if l_epoch_count == l_epoch - 1:
            l_epoch_check = True
            w_locals_server.append(copy.deepcopy(w_server))
            w_locals_server_tier[client_tier[idx]].append(copy.deepcopy(w_server))
            
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
            
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            if idx not in idx_collect:
                idx_collect.append(idx)
        
        # Federation process
        if len(idx_collect) == m:
            fed_check = True
            
            w_locals_tier = w_locals_server
            w_locals_server = []
            w_locals_server_tier = {}
            for i in range(1, num_tiers+1):
                w_locals_server_tier[i] = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
            logger.info(f"Server LR: {optimizer_server.param_groups[0]['lr']}")
            new_lr = optimizer_server.param_groups[0]['lr']
    
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server, net_glob_server_tier
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check, w_glob_server_tier
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, acc_avg_all_user, loss_avg_all_user_train, best_acc
    global wait, new_lr
    
    net = copy.deepcopy(net_model_server_tier[idx]).to(device)
    net.eval()
    
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        
        # forward prop
        fx_server = net(fx_client)
        
        # calculate loss
        y = y.to(torch.long)
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Global Model Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(acc_avg_test, loss_avg_test))
            
            if l_epoch_check:
                l_epoch_check = False
                
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
            
            if fed_check:
                fed_check = False
                logger.info("------------------------------------------------")
                logger.info("------ Federation process at Server-Side ------- ")
                logger.info("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)
                
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []
                
                if (acc_avg_all_user / 100) > best_acc * (1 + DYNAMIC_LR_THRESHOLD):
                    logger.info("- Found better accuracy")
                    best_acc = (acc_avg_all_user / 100)
                    wait = 0
                else:
                    wait += 1
                    logger.info(f'wait {wait}')
                if wait > patience:
                    new_lr = max(float(optimizer_server.param_groups[0]['lr']) * factor, min_lr)
                    wait = 0
                
                logger.info("==========================================================")
                logger.info("{:^58}".format("DTFL Performance"))
                logger.info("----------------------------------------------------------")
                logger.info(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                logger.info(' Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                logger.info("==========================================================")
    
    return


# Client-side class
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = client_epoch[idx]
        self.ldr_train = dataset_train[idx]
        self.ldr_test = dataset_test[idx]
    
    def train(self, net):
        net.train()
        self.lr, lr = new_lr, new_lr
        
        if args.optimizer == "Adam":
            optimizer_client = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True)
        elif args.optimizer == "SGD":
            optimizer_client = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                              nesterov=True,
                                              weight_decay=args.wd)
        
        time_client = 0
        client_intermediate_data_size = 0
        CEloss_client_train = []
        Dcorloss_client_train = []
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                # forward prop
                extracted_features, fx = net(images)
                
                if args.PatchShuffle == 1:
                    fx_shuffled = fx.clone().detach().requires_grad_(False)
                    fx_shuffled = PatchShuffle(fx_shuffled)
                    client_fx = fx_shuffled.clone().detach().requires_grad_(True)
                else:
                    client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                time_client += time.time() - time_s
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, _)
                
                # backward prop
                time_s = time.time()
                
                labels = labels.to(torch.long)
                loss = criterion(extracted_features, labels)
                CEloss_client_train.append(((1 - dcor_coefficient) * loss.item()))
                
                if whether_dcor:
                    Dcor_value = dis_corr(images, fx)
                    loss = (1 - dcor_coefficient) * loss + dcor_coefficient * Dcor_value
                    Dcorloss_client_train.append(((dcor_coefficient) * Dcor_value))
                
                loss.backward()
                optimizer_client.step()
                time_client += time.time() - time_s
                
                client_intermediate_data_size += (sys.getsizeof(client_fx.storage()) +
                                                  sys.getsizeof(labels.storage()))
        
        global intermediate_data_size
        intermediate_data_size += client_intermediate_data_size
        
        return net.state_dict(), time_client, client_intermediate_data_size
    
    def evaluate(self, net, ell):
        net.eval()
        
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                
                extracted_features, fx = net(images)
                evaluate_server(fx, labels, self.idx, len_batch, ell)
        
        return


def calculate_client_samples(train_data_local_num_dict, idxs_users, dataset):
    """Calculate the number of samples for each client"""
    num_users = len(idxs_users)
    client_sample = []
    total_samples = sum(train_data_local_num_dict.values())
    for idx in idxs_users:
        client_sample.append(train_data_local_num_dict[idx] / total_samples * num_users)
    return client_sample


def create_balanced_test_dataset(args):
    """Create balanced test dataset for generalization evaluation"""
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                [0.2673, 0.2564, 0.2762])
        ])
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "fashion_mnist":
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.2860, 0.2860, 0.2860], [0.3530, 0.3530, 0.3530])
        ])
        testset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return test_loader


def evaluate_dtfl_generalization(args, iid_test_loader, net_glob_client_tier, net_glob_server_tier, device):
    """Evaluate DTFL model generalization on balanced test set"""
    logger.info("\n===== Evaluating DTFL Generalization =====")
    
    results = {}
    
    for tier in range(1, 8):
        if tier in net_glob_client_tier and tier in net_glob_server_tier:
            client_model = copy.deepcopy(net_glob_client_tier[tier]).to(device)
            server_model = copy.deepcopy(net_glob_server_tier[tier]).to(device)
            
            client_model.eval()
            server_model.eval()
            
            correct = 0
            total = 0
            class_correct = [0] * 10
            class_total = [0] * 10
            
            with torch.no_grad():
                for data, target in iid_test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    extracted_features, fx = client_model(data)
                    fx_server = server_model(fx)
                    
                    _, predicted = fx_server.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    for i in range(len(target)):
                        label = target[i].item()
                        if label < len(class_correct):
                            class_total[label] += 1
                            if predicted[i] == label:
                                class_correct[label] += 1
            
            accuracy = 100.0 * correct / total if total > 0 else 0
            class_acc = [100.0 * c / max(1, t) for c, t in zip(class_correct, class_total)]
            
            results[tier] = {
                "accuracy": accuracy,
                "class_acc": class_acc
            }
            
            logger.info(f"Tier {tier} - Balanced Test Accuracy: {accuracy:.2f}%")
    
    if results:
        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
        logger.info(f"\nAverage Balanced Test Accuracy: {avg_accuracy:.2f}%")
    else:
        avg_accuracy = 0
        logger.info("\nNo models available for evaluation")
    
    return results


def evaluate_split_models(net_glob_client_tier, net_glob_server_tier, client_tier, dataset_test, device, num_users):
    """Evaluate all client-server split models"""
    logger.info("\n===== Evaluating Split Models =====")
    
    all_accuracies = []
    
    for client_idx in range(num_users):
        tier = client_tier[client_idx]
        
        client_model = copy.deepcopy(net_glob_client_tier[tier]).to(device)
        server_model = copy.deepcopy(net_glob_server_tier[tier]).to(device)
        
        client_model.eval()
        server_model.eval()
        
        test_loader = dataset_test[client_idx]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                try:
                    extracted_features, fx = client_model(data)
                    fx_server = server_model(fx)
                    
                    _, predicted = fx_server.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error evaluating client {client_idx}: {str(e)}")
                    continue
        
        if total > 0:
            accuracy = 100.0 * correct / total
            all_accuracies.append(accuracy)
            logger.info(f"Client {client_idx} (Tier {tier}) Split Model Accuracy: {accuracy:.2f}%")
    
    if all_accuracies:
        avg_accuracy = sum(all_accuracies) / len(all_accuracies)
        logger.info(f"Average Split Model Accuracy: {avg_accuracy:.2f}%")
        return avg_accuracy
    else:
        logger.info("No split models evaluated successfully")
        return 0.0


# Training initialization
net_glob_client.train()
w_glob_client_tier = {}

for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()

w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)

w_glob = copy.deepcopy(init_glob_model.state_dict())

for t in range(1, num_tiers+1):
    for k in w_glob_client_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:]
        
        if (k1 == 'fc.bias' or k1 == 'fc.weight'):
            continue
        
        w_glob_client_tier[t][k] = w_glob[k1]
    
    for k in w_glob_server_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:]
        w_glob_server_tier[t][k] = w_glob[k1]
    
    net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
    net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])

w_locals_tier, w_locals_client, w_locals_server = [], [], []

net_model_client_tier = {}
for i in range(1, num_tiers+1):
    net_model_client_tier[i] = net_glob_client_tier[i]
    net_model_client_tier[i].train()

for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()

optimizer_client_tier = {}
for i in range(0, num_users):
    if args.optimizer == "Adam":
        optimizer_client_tier[i] = torch.optim.Adam(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, weight_decay=args.wd, amsgrad=True)
    elif args.optimizer == "SGD":
        optimizer_client_tier[i] = torch.optim.SGD(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, momentum=0.9,
                                                    nesterov=True,
                                                    weight_decay=args.wd)

# ============================================================================
# Execute Warmup Phase
# ============================================================================
warmed_up_models = warmup_phase(args, net_glob_client_tier, client_tier,
                                dataset_train, device, num_users)

# Load warmed-up models into client models
if warmed_up_models is not None:
    logger.info("Loading warmed-up models into global client models...")
    for client_idx in range(num_users):
        client_model_tier = client_tier[client_idx]
        net_glob_client_tier[client_model_tier].load_state_dict(warmed_up_models[client_idx])
    
    # Update global client weights after warmup
    for i in range(1, num_tiers+1):
        w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()
    
    logger.info("Warmed-up models loaded successfully")
else:
    logger.info("No warmup performed, using initialized models")

# ============================================================================
# Main Federated Training Loop
# ============================================================================
logger.info("\n" + "="*80)
logger.info("Starting Federated Training")
logger.info("="*80)

T_max = 1000  # Initialize T_max
total_time = 0
avg_tier_time_list = []
max_time_list = pd.DataFrame({'time': []})

client_delay_computing = 0.1
client_delay_net = 0.1

simulated_delay_historical_df = pd.DataFrame()
start_time = time.time()

client_observed_times = pd.DataFrame()
torch.manual_seed(SEED)
simulated_delay = np.zeros(num_users)

idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)

data_transmitted_client_all = {}

computation_time_clients = {}
for k in range(num_users):
    computation_time_clients[k] = []

# Create balanced test loader for generalization evaluation
logger.info("Creating balanced test dataset for generalization evaluation...")
balanced_test_loader = create_balanced_test_dataset(args)

client_tier_all = []
client_tier_all.append(copy.deepcopy(client_tier))
total_training_time = 0
time_train_server_train_all_list = []

client_sample = np.ones(num_users)

# Main training loop
for iter in range(epochs):
    logger.info(f"\n{'='*80}")
    logger.info(f"Round {iter+1}/{epochs}")
    logger.info(f"{'='*80}")

    split_avg_accuracy = evaluate_split_models(
        net_glob_client_tier,
        net_glob_server_tier,
        client_tier,
        dataset_test,
        device,
        num_users
    )
    
    w_locals_client = []
    w_locals_client_tier = {}
    w_locals_client_tier = {i: [] for i in range(1, num_tiers+1)}
    
    client_observed_time = np.zeros(num_users)
    processes = []
    simulated_delay = np.zeros(num_users)
    
    for idx in idxs_users:
        data_server_to_client = calculate_data_size(w_glob_client_tier[client_tier[idx]])
        simulated_delay[idx] = data_server_to_client / net_speed[idx]
        
        client_model_parameter_data_size = 0
        time_train_test_s = time.time()
        net_glob_client = net_model_client_tier[client_tier[idx]]
        w_glob_client_tier[client_tier[idx]] = net_glob_client_tier[client_tier[idx]].state_dict()
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test, idxs=[], idxs_test=[])
        
        # Training
        [w_client, duration, client_intermediate_data_size] = local.train(net=copy.deepcopy(net_glob_client).to(device))
        
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_tier[client_tier[idx]].append(copy.deepcopy(w_client))
        
        # Testing
        if idx == idxs_users[-1]:
            net = copy.deepcopy(net_glob_client)
            w_previous = copy.deepcopy(net.state_dict())
            net.load_state_dict(w_client)
            net.to(device)
            
            local.evaluate(net, ell=iter)
            net.load_state_dict(w_previous)
        
        client_observed_time[idx] = duration
        
        client_model_parameter_data_size = calculate_data_size(w_client)
        model_parameter_data_size += client_model_parameter_data_size
        
        data_transmitted_client = client_intermediate_data_size + client_model_parameter_data_size
        data_transmitted_client_all[idx] = data_transmitted_client
        
        simulated_delay[idx] += compute_delay(data_transmitted_client, net_speed[idx],
                                              delay_coefficient[idx], duration)
    
    # Evaluate generalization every 5 rounds or last round
    if iter % 5 == 0 or iter == epochs - 1:
        logger.info(f"\n===== Round {iter+1} Generalization Evaluation =====")
        dtfl_generalization_results = evaluate_dtfl_generalization(
            args,
            balanced_test_loader,
            net_glob_client_tier,
            net_glob_server_tier,
            device
        )
        
        if dtfl_generalization_results:
            balanced_avg_acc = sum(r["accuracy"] for r in dtfl_generalization_results.values()) / len(dtfl_generalization_results)
            logger.info(f"DTFL Average Balanced Test Accuracy: {balanced_avg_acc:.2f}%")
    
    server_wait_first_to_last_client = (max(simulated_delay * client_epoch) - min(simulated_delay * client_epoch))
    training_time = (max(simulated_delay))
    total_training_time += training_time
    if iter == 0:
        first_training_time = training_time
    times_in_server = []
    time_train_server_train_all_list.append(time_train_server_train_all)
    time_train_server_train_all = 0
    
    simulated_delay[simulated_delay == 0] = np.nan
    simulated_delay_historical_df = pd.concat([simulated_delay_historical_df, pd.DataFrame(simulated_delay).T], ignore_index=True)
    client_observed_times = pd.concat([client_observed_times, pd.DataFrame(client_observed_time).T], ignore_index=True)
    client_epoch_last = client_epoch.copy()
    
    idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)
    
    [client_tier, T_max, computation_time_clients] = TierScheduler(computation_time_clients, T_max, client_tier_all=client_tier_all,
                                                delay_history=simulated_delay_historical_df,
                                                num_tiers=num_tiers, client_epoch=client_epoch,
                                                num_users=num_users, dataset_size=dataset_size,
                                                batch_size=args.batch_size,
                                                data_transmitted_client_all=data_transmitted_client_all,
                                                net_speed=net_speed)
    
    client_tier_all.append(copy.deepcopy(client_tier))
    
    for i in client_tier.keys():
        net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]
    
    # Federation process
    logger.info("-----------------------------------------------------------")
    logger.info("{:^59}".format("Model Aggregation"))
    logger.info("-----------------------------------------------------------")
    
    client_sample = calculate_client_samples(train_data_local_num_dict, idxs_users, args.dataset)
    
    w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users)
    
    for t in range(1, num_tiers+1):
        for k in w_glob_client_tier[t].keys():
            if k in w_glob_server_tier[t].keys():
                if w_locals_client_tier[t] != []:
                    w_glob_client_tier[t][k] = FedAvg(w_locals_client_tier[t])[k]
                    continue
                else:
                    continue
            
            w_glob_client_tier[t][k] = w_glob[k]
        
        for k in w_glob_server_tier[t].keys():
            w_glob_server_tier[t][k] = w_glob[k]
        
        net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
        net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
    
    logger.info(f'Size of Total Model Parameter Data Transferred {(model_parameter_data_size/1024**2):,.2f} MB')
    logger.info(f'Size of Total Intermediate Data Transferred {(intermediate_data_size/1024**2):,.2f} MB')

elapsed = (time.time() - start_time) / 60

logger.info("\n" + "="*80)
logger.info("Training and Evaluation completed!")
logger.info(f"Total elapsed time: {elapsed:.2f} minutes")
logger.info("="*80)
