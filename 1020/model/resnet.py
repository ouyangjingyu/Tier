import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_resnet_config(model_type='resnet56'):
    """根据模型类型获取ResNet配置"""
    if model_type == 'resnet56':
        # ResNet-56: 每层9个块，总共27个块
        num_blocks = [9, 9, 9]  # 三层的块数
        # 客户端：第一层全部(9) + 第二层前半部分(4)，总共13个块
        client_blocks = [9, 4]  
        # 服务器：第二层后半部分(5) + 第三层全部(9)，总共14个块
        server_blocks = [5, 9]  
        client_output_channels = 32  # 客户端输出通道数
        server_input_channels = 32   # 服务器输入通道数
    elif model_type == 'resnet110':
        # ResNet-110: 每层18个块，总共54个块
        num_blocks = [18, 18, 18]
        # 客户端：第一层全部(18) + 第二层前半部分(9)，总共27个块
        client_blocks = [18, 9]  
        # 服务器：第二层后半部分(9) + 第三层全部(18)，总共27个块
        server_blocks = [9, 18]  
        client_output_channels = 32
        server_input_channels = 32
    else:
        # 默认使用ResNet-56配置
        num_blocks = [9, 9, 9]
        client_blocks = [9, 4]
        server_blocks = [5, 9]
        client_output_channels = 32
        server_input_channels = 32
    
    return num_blocks, client_blocks, server_blocks, client_output_channels, server_input_channels


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TierAwareClientModel(nn.Module):
    """客户端模型：包含ResNet前半部分作为共享层 + 个性化路径"""
    def __init__(self, num_classes=10, tier=1, model_type='resnet56', input_channels=3):
        super(TierAwareClientModel, self).__init__()
        self.tier = tier
        self.model_type = model_type
        self.input_channels = input_channels
        
        # 获取模型配置
        _, client_blocks, _, client_output_channels, _ = get_resnet_config(model_type)
        
        # 共享层 - ResNet前半部分，所有客户端完全一致
        self.shared_base = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Layer1 - 全部块
            self._make_layer(BasicBlock, 16, 16, client_blocks[0]),
            # Layer2 - 前半部分块
            self._make_layer(BasicBlock, 16, 32, client_blocks[1], stride=2)
        )
        
        # 共享层输出通道数
        self.output_channels = client_output_channels
        
        # 根据tier调整个性化路径
        self._create_personalized_path(num_classes)
        
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        
        return nn.Sequential(*layers)
    
    def _create_personalized_path(self, num_classes):
        """根据tier创建个性化路径"""
        if self.tier == 1:  # 高性能设备
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 32, 2),
                self._make_layer(BasicBlock, 32, 64, 2, stride=2)
            )
            local_dim = 64
        elif self.tier == 2:  # 中高性能设备
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 32, 2),
                self._make_layer(BasicBlock, 32, 64, 1, stride=2)
            )
            local_dim = 64
        elif self.tier == 3:  # 中低性能设备
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 32, 1),
                self._make_layer(BasicBlock, 32, 64, 1, stride=2)
            )
            local_dim = 64
        else:  # tier 4, 低性能设备
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 64, 1, stride=2)
            )
            local_dim = 64
        
        # 本地分类器 - 简化为单层FC
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(local_dim, num_classes)
        )
    
    def forward(self, x):
        # 共享基础层（输出给服务器）
        shared_features = self.shared_base(x)
        
        # 个性化路径
        personal_features = self.personalized_path(shared_features)
        
        # 本地分类
        local_logits = self.local_classifier(personal_features)
        
        return local_logits, shared_features, personal_features
    
    def get_shared_params(self):
        """获取共享层参数"""
        shared_params = {}
        for name, param in self.named_parameters():
            if 'shared_base' in name:
                shared_params[name] = param
        return shared_params
    
    def get_personalized_params(self):
        """获取个性化路径参数"""
        personalized_params = {}
        for name, param in self.named_parameters():
            if 'personalized_path' in name or 'local_classifier' in name:
                personalized_params[name] = param
        return personalized_params


class EnhancedServerModel(nn.Module):
    """服务器模型：ResNet后半部分 + 分类器"""
    def __init__(self, num_classes=10, model_type='resnet56', input_channels=3):
        super(EnhancedServerModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 获取模型配置
        _, _, server_blocks, _, server_input_channels = get_resnet_config(model_type)
        
        # 服务器层 - ResNet后半部分
        self.server_layers = nn.Sequential(
            # Layer2剩余部分 - 从32通道继续
            self._make_layer(BasicBlock, server_input_channels, 32, server_blocks[0]),
            # Layer3 - 完整层
            self._make_layer(BasicBlock, 32, 64, server_blocks[1], stride=2)
        )
        
        # 分类器 - 简化为单层FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 服务器层处理
        x = self.server_layers(x)
        
        # 分类器输出最终结果
        logits = self.classifier(x)
        
        return logits
        
    def get_params(self):
        """获取所有参数"""
        return {name: param for name, param in self.named_parameters()}


# 保留原有的ImprovedGlobalClassifier以兼容性（虽然不再使用）
class ImprovedGlobalClassifier(nn.Module):
    """改进的全局分类器（已废弃，保留兼容性）"""
    def __init__(self, feature_dim=128, num_classes=10):
        super(ImprovedGlobalClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)
    
    def get_params(self):
        return {name: param for name, param in self.named_parameters()}