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


def get_resnet_num_blocks(model_type='resnet56'):
    num_blocks, _, _, _, _ = get_resnet_config(model_type)
    return num_blocks


def get_tier_shared_block_counts(model_type='resnet56', tier=1):
    num_blocks = get_resnet_num_blocks(model_type)
    l1, l2, l3 = num_blocks

    if model_type == 'resnet110':
        tier_map = {
            1: (l1, l2, 9),
            2: (l1, l2, 6),
            3: (l1, l2, 3),
            4: (l1, l2, 0),
            5: (l1, 14, 0),
            6: (l1, 9, 0),
            7: (l1, 0, 0),
        }
    else:
        tier_map = {
            1: (l1, l2, 6),
            2: (l1, l2, 3),
            3: (l1, l2, 0),
            4: (l1, 8, 0),
            5: (l1, 6, 0),
            6: (l1, 4, 0),
            7: (l1, 0, 0),
        }

    shared_l1, shared_l2, shared_l3 = tier_map.get(int(tier), tier_map[4])
    shared_l1 = max(0, min(l1, int(shared_l1)))
    shared_l2 = max(0, min(l2, int(shared_l2)))
    if shared_l2 == 0:
        shared_l3 = 0
    if shared_l2 < l2:
        shared_l3 = 0
    shared_l3 = max(0, min(l3, int(shared_l3)))
    return {'layer1': shared_l1, 'layer2': shared_l2, 'layer3': shared_l3}


def _layer_block_specs(layer_name, num_blocks):
    if layer_name == 'layer1':
        return [(16, 16, 1) for _ in range(num_blocks)]
    if layer_name == 'layer2':
        if num_blocks <= 0:
            return []
        return [(16, 32, 2)] + [(32, 32, 1) for _ in range(num_blocks - 1)]
    if layer_name == 'layer3':
        if num_blocks <= 0:
            return []
        return [(32, 64, 2)] + [(64, 64, 1) for _ in range(num_blocks - 1)]
    raise ValueError(f"Unknown layer name: {layer_name}")


class BlockDict(nn.Module):
    def __init__(self, blocks=None):
        super().__init__()
        self.blocks = nn.ModuleDict(blocks or {})

    def forward(self, x):
        for idx in sorted(self.blocks.keys(), key=lambda k: int(k)):
            x = self.blocks[idx](x)
        return x

    def keys(self):
        return self.blocks.keys()


class StablePaddedClassifier(nn.Module):
    def __init__(self, num_classes, max_channels=64):
        super().__init__()
        self.max_channels = int(max_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.max_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if x.size(1) < self.max_channels:
            pad = x.new_zeros(x.size(0), self.max_channels - x.size(1))
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > self.max_channels:
            x = x[:, : self.max_channels]
        return self.fc(x)


def _build_blocks(layer_name, total_blocks, include_indices):
    specs = _layer_block_specs(layer_name, total_blocks)
    blocks = {}
    for i in include_indices:
        inplanes, planes, stride = specs[i]
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        blocks[str(i)] = BasicBlock(inplanes, planes, stride=stride, downsample=downsample)
    return BlockDict(blocks)


class TierSharedBackbone(nn.Module):
    def __init__(self, tier, model_type='resnet56', input_channels=3):
        super().__init__()
        self.tier = int(tier)
        self.model_type = model_type
        self.input_channels = input_channels

        num_blocks = get_resnet_num_blocks(model_type)
        shared = get_tier_shared_block_counts(model_type=model_type, tier=tier)

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = _build_blocks('layer1', num_blocks[0], range(shared['layer1']))
        self.layer2 = _build_blocks('layer2', num_blocks[1], range(shared['layer2']))
        self.layer3 = _build_blocks('layer3', num_blocks[2], range(shared['layer3']))

        if shared['layer3'] > 0:
            self.output_channels = 64
        elif shared['layer2'] > 0:
            self.output_channels = 32
        else:
            self.output_channels = 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TierAwareClientModel(nn.Module):
    """客户端模型：按tier异构共享层 + 共享层变化仍稳定的本地分类器"""
    def __init__(self, num_classes=10, tier=1, model_type='resnet56', input_channels=3):
        super(TierAwareClientModel, self).__init__()
        self.tier = int(tier)
        self.model_type = model_type
        self.input_channels = input_channels
        self.shared_base = TierSharedBackbone(tier=self.tier, model_type=model_type, input_channels=input_channels)
        self.local_classifier = StablePaddedClassifier(num_classes=num_classes, max_channels=64)
    
    def forward(self, x):
        shared_features = self.shared_base(x)
        local_logits = self.local_classifier(shared_features)
        return local_logits, shared_features, shared_features
    
    def get_shared_params(self):
        """获取共享层参数"""
        shared_params = {}
        for name, param in self.named_parameters():
            if 'shared_base' in name:
                shared_params[name] = param
        return shared_params
    
    def update_tier(self, new_tier):
        new_tier = int(new_tier)
        if new_tier == self.tier:
            return
        old_state = self.shared_base.state_dict()
        self.tier = new_tier
        self.shared_base = TierSharedBackbone(tier=self.tier, model_type=self.model_type, input_channels=self.input_channels)
        self.shared_base.load_state_dict(old_state, strict=False)


class EnhancedServerModel(nn.Module):
    """服务器模型：ResNet后半部分 + 分类器"""
    def __init__(self, num_classes=10, tier=1, model_type='resnet56', input_channels=3):
        super(EnhancedServerModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.tier = int(tier)
        self.model_type = model_type
        
        num_blocks = get_resnet_num_blocks(model_type)
        shared = get_tier_shared_block_counts(model_type=model_type, tier=tier)

        self.layer1 = _build_blocks('layer1', num_blocks[0], range(shared['layer1'], num_blocks[0]))
        self.layer2 = _build_blocks('layer2', num_blocks[1], range(shared['layer2'], num_blocks[1]))
        self.layer3 = _build_blocks('layer3', num_blocks[2], range(shared['layer3'], num_blocks[2]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
        
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
