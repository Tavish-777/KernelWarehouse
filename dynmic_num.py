import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import math
import json
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import DropPath
import random
from itertools import product

# --- 0. 辅助函数 ---

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def legendre_basis(H, W, k, device):
    """
    生成一个更复杂的、包含交叉阶数的勒让德多项式基的傅里叶变换结果。
    这是经过优化的版本，会预计算一维多项式。
    """
    if k == 0:
        return torch.empty(0, H, W // 2 + 1, dtype=torch.cfloat, device=device)

    # 为了选出k个，生成一个稍大的候选池
    max_total_order = int(np.sqrt(k) * 1.5) + 2
    i_orders, j_orders = range(max_total_order), range(max_total_order)
    order_pairs = sorted(list(product(i_orders, j_orders)), key=lambda p: (p[0] + p[1], max(p[0], p[1])))[:k]
    
    max_order = max(max(p) for p in order_pairs) if order_pairs else -1

    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    
    # 预计算一维多项式
    leg_polys_y = [torch.from_numpy(np.polynomial.legendre.legval(y_coords.cpu().numpy(), [0]*i+[1])).to(device).float() for i in range(max_order + 1)]
    leg_polys_x = [torch.from_numpy(np.polynomial.legendre.legval(x_coords.cpu().numpy(), [0]*i+[1])).to(device).float() for i in range(max_order + 1)]
    
    basis_list = []
    for i, j in order_pairs:
        basis_2d = torch.outer(leg_polys_y[i], leg_polys_x[j])
        basis_ft = torch.fft.rfft2(basis_2d, s=(H, W), norm='ortho')
        basis_list.append(basis_ft)
        
    return torch.stack(basis_list, dim=0)

# --- 1. 频域处理核心模块 (全新、简洁的设计) ---

class FrequencyBasisManager(nn.Module):
    """
    一个集中的管理器，用于创建、存储和共享不同尺寸的傅里-勒让德基。
    """
    def __init__(self):
        super().__init__()
        # 这个管理器是无参数的，它只管理 buffer

    def get_or_create_fourier_basis(self, H, W, k, device):
        """
        获取或创建指定尺寸的傅里叶基。基被缓存为模块的buffer，以实现共享。
        """
        buffer_name = f'fourier_basis_{H}x{W}_k{k}'
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)
        else:
            print(f"INFO: FrequencyBasisManager creating shared Buffer '{buffer_name}' with {k}+1 bases on device {device}...")
            legendre_bases = legendre_basis(H, W, k, device=device)
            zero_basis = torch.zeros(1, H, W // 2 + 1, dtype=torch.cfloat, device=device)
            final_basis = torch.cat([legendre_bases, zero_basis], dim=0)
            self.register_buffer(buffer_name, final_basis)
            return final_basis

class FrequencyAttention(nn.Module):
    """为频域路径生成动态系数。"""
    def __init__(self, in_planes, num_fourier_basis, num_frequency_mixtures, reduction=0.0625):
        super().__init__()
        hidden_planes = max(int(in_planes * reduction), 16)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_planes, hidden_planes)
        self.norm1 = nn.LayerNorm(hidden_planes)
        self.relu = nn.ReLU(inplace=True)
        
        fourier_head_planes = in_planes * num_frequency_mixtures * (num_fourier_basis + 1)
        self.fourier_head = nn.Linear(hidden_planes, fourier_head_planes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.shape
        context = self.avgpool(x).view(b, c)
        context = self.relu(self.norm1(self.fc1(context)))
        return self.fourier_head(context)

class FrequencyBranch(nn.Module):
    """
    动态频域滤波支路，实现了“插值工作区”的逻辑。
    """
    def __init__(self, in_planes, out_planes, stride, basis_manager, num_fourier_basis, num_frequency_mixtures=16):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.basis_manager = basis_manager
        self.num_fourier_basis = num_fourier_basis
        self.num_frequency_mixtures = num_frequency_mixtures

        self.attention = FrequencyAttention(in_planes, self.num_fourier_basis, num_frequency_mixtures)
        self.pointwise_mixer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pointwise_mixer_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        B, C_in, H, W = x.shape
        device = x.device

        kw_attention_fourier = self.attention(x)

        mix_h = mix_w = int(np.sqrt(self.num_frequency_mixtures))
        H_out, W_out = H // self.stride[0], W // self.stride[1]
        
        # 1. 计算并插值到规整的工作区
        H_work = math.ceil(H / mix_h) * mix_h
        # W_work 的计算需要从频域反推
        W_f = W // 2 + 1
        W_f_work = math.ceil(W_f / mix_w) * mix_w
        W_work = (W_f_work - 1) * 2 if W_f_work > 1 else 1

        if (H_work, W_work) != (H, W):
            x_work = F.interpolate(x, size=(H_work, W_work), mode='bilinear', align_corners=False)
        else:
            x_work = x

        # 2. 在工作区内进行频域操作
        fourier_basis_work = self.basis_manager.get_or_create_fourier_basis(H_work, W_work, self.num_fourier_basis, device)
        x_ft_work = torch.fft.rfft2(x_work, s=(H_work, W_work), norm='ortho')
        
        total_basis_count = self.num_fourier_basis + 1
        coefficients_raw = kw_attention_fourier.reshape(B, C_in, self.num_frequency_mixtures, total_basis_count)
        coefficients = torch.softmax(coefficients_raw, dim=-1)
        coefficients = torch.complex(coefficients, torch.zeros_like(coefficients))

        patch_h = H_work // mix_h
        patch_w_f = x_ft_work.shape[-1] // mix_w
        
        # 分块
        basis_reshaped = fourier_basis_work.reshape(total_basis_count, mix_h, patch_h, mix_w, patch_w_f)
        basis_permuted = basis_reshaped.permute(1, 3, 0, 2, 4).contiguous()
        basis_patched = basis_permuted.reshape(self.num_frequency_mixtures, total_basis_count, patch_h, patch_w_f)
        
        x_ft_reshaped = x_ft_work.reshape(B, C_in, mix_h, patch_h, mix_w, patch_w_f)
        x_ft_permuted = x_ft_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_ft_patched = x_ft_permuted.reshape(B, self.num_frequency_mixtures, C_in, patch_h, patch_w_f)

        # 3. 合成与滤波
        coeffs_permuted = coefficients.permute(0, 2, 1, 3) # -> [B, mix, i, k]
        kernel_patched = torch.einsum("bmik,mk...->bmi...", coeffs_permuted, basis_patched)
        output_ft_patched = x_ft_patched * kernel_patched

        # 4. 拼接与逆变换
        output_ft_padded = output_ft_patched.reshape(B, mix_h, mix_w, C_in, patch_h, patch_w_f).permute(0, 3, 1, 4, 2, 5).contiguous()
        output_ft_dw = output_ft_padded.reshape(B, C_in, H_work, x_ft_work.shape[-1])
        
        target_s_work = (H_work // self.stride[0], W_work // self.stride[1])
        output_fourier_dw_work = torch.fft.irfft2(output_ft_dw, s=target_s_work, norm='ortho')

        # 5. 尺寸恢复和最终处理
        if output_fourier_dw_work.shape[2:] != (H_out, W_out):
            output_fourier_dw = F.interpolate(output_fourier_dw_work, size=(H_out, W_out), mode='bilinear', align_corners=False)
        else:
            output_fourier_dw = output_fourier_dw_work
            
        output_fourier = self.pointwise_mixer_bn(self.pointwise_mixer(output_fourier_dw))
        
        return output_fourier

# --- 2. 混合卷积模块和ResNet模型定义 ---

class HybridConv(nn.Module):
    """混合卷积层。"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, basis_manager=None, num_fourier_basis=16):
        super().__init__()
        self.conv_spatial = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias)
        self.freq_branch = FrequencyBranch(in_planes, out_planes, stride=(stride, stride), basis_manager=basis_manager, num_fourier_basis=num_fourier_basis)
        self.fourier_path_gamma = nn.Parameter(torch.zeros(1, out_planes, 1, 1))

    def forward(self, x):
        out_spatial = self.conv_spatial(x)
        out_freq = self.freq_branch(x)
        if out_spatial.shape != out_freq.shape:
             out_freq = F.interpolate(out_freq, size=out_spatial.shape[2:], mode='bilinear', align_corners=False)
        return out_spatial + self.fourier_path_gamma * out_freq

class HybridBasicBlock(nn.Module):
    """使用HybridConv构建的ResNet基本残差块。"""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, basis_manager=None, drop_path=0., num_fourier_basis=16):
        super().__init__()
        self.conv1 = HybridConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, basis_manager=basis_manager, num_fourier_basis=num_fourier_basis)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = HybridConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, basis_manager=basis_manager, num_fourier_basis=num_fourier_basis)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out) + self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetHybrid(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_fourier_basis=[8, 16, 24, 32], drop_path_rate=0.):
        super().__init__()
        self.in_planes = 64

        # 创建一个集中的、单一的傅里叶基管理器
        self.basis_manager = FrequencyBasisManager()

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, dpr=dpr[:sum(num_blocks[:1])], nfb=num_fourier_basis[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dpr=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])], nfb=num_fourier_basis[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dpr=dpr[sum(num_blocks[:2]):sum(num_blocks[:3])], nfb=num_fourier_basis[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dpr=dpr[sum(num_blocks[:3]):], nfb=num_fourier_basis[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dpr, nfb):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            layers.append(block(self.in_planes, planes, s, basis_manager=self.basis_manager, drop_path=dpr[i], num_fourier_basis=nfb))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # ... (与之前完全相同的 forward 流程) ...
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18Hybrid(num_classes=10, num_fourier_basis=[8, 16, 24, 32], drop_path_rate=0.):
    """工厂函数，用于创建ResNet-18混合模型。"""
    return ResNetHybrid(HybridBasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_fourier_basis=num_fourier_basis, drop_path_rate=drop_path_rate)

# --- 3. 训练与评估 ---

def train(epoch, model, trainloader, optimizer, criterion, device, mixup_fn=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 使用tqdm添加进度条
    pbar = tqdm(trainloader, desc=f"Epoch {epoch:3d} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 应用Mixup/Cutmix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        # 处理来自Mixup的硬标签与软标签的准确率计算
        if targets.ndim == 2: # 来自Mixup的软标签
            correct += predicted.eq(targets.argmax(dim=1)).sum().item()
        else: # 硬标签
            correct += predicted.eq(targets).sum().item()
        # 更新tqdm进度条的后缀信息
        pbar.set_postfix(loss=f"{train_loss/(batch_idx+1):.3f}", acc=f"{100.*correct/total:.2f}%")

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total
    return avg_loss, acc

def test(model, testloader, device):
    # 仿照 engine.py 的写法，评估函数应该使用自己的标准损失函数，因为它处理的是未经修改的“硬”标签。
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # 使用tqdm添加进度条
    pbar = tqdm(testloader, desc="          [Test] ")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新tqdm进度条的后缀信息
            pbar.set_postfix(loss=f"{test_loss/(batch_idx+1):.3f}", acc=f"{100.*correct/total:.2f}%")
    
    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    return avg_loss, acc

def get_args_parser():
    parser = argparse.ArgumentParser('ResNet-Hybrid training and evaluation script', add_help=False)
    
    # --- Essential parameters ---
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='runs/resnet_hybrid_cifar100_timm_recipe',
                        help='path where to save logs and checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # --- Training hyperparameters ---
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')

    # --- Augmentation and Regularization ---
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # --- Model parameters ---
    parser.add_argument('--model', default='resnet18_hybrid', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_fourier_basis', type=int, nargs='+', default=[16,16,64,64],
                        help='Number of fourier basis for each of the 4 stages.')

    return parser

# --- 4. 主程序 ---
def main(args):
    print("==> Preparing data...")
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- 固定随机种子以保证可复现性 ---
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # 适用于多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"==> Random seed set to {args.seed}")
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    # --- 日志和模型输出目录 ---
    log_path = os.path.join(args.output_dir, "log.txt")

    # --- 采用 main.py 风格的数据增强 ---
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4), # 添加ColorJitter
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        transforms.RandomErasing(p=0.25),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(int(args.input_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    # 关键点：在Windows上，num_workers > 0 必须在 if __name__ == '__main__': 保护的代码块中
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, 
        batch_size=int(args.batch_size * 1.5), 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print('==> Building model..')
    # 实例化新的混合模型，并将类别数改为100
    model = ResNet18Hybrid(
        num_classes=args.nb_classes, 
        num_fourier_basis=args.num_fourier_basis,
        drop_path_rate=args.drop_path
    ).to(device)
    
    # 打印模型结构，可以看到 HybridConv 模块
    print(model)

    # --- 采用 main.py 的训练组件 ---
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"==> Using criterion: {criterion.__class__.__name__}")
    freq_params = [p for n, p in model.named_parameters() if 'fourier_path_gamma' in n]
    base_params = [p for n, p in model.named_parameters() if 'fourier_path_gamma' not in n]

    optimizer = optim.AdamW([
        {'params': base_params},
        {'params': freq_params, 'lr': args.lr * 10} # 给 gamma 10倍的学习率
    ], lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("==> Starting Training...")
    max_accuracy = 0.0

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, device, mixup_fn)
        test_loss, test_acc = test(model, testloader, device)

        scheduler.step()
        epoch_time = time.time() - start_time

        # 打印每轮的总结
        print(f"Epoch {epoch:3d} Summary | Train Loss: {train_loss:.3f} | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

        # 日志记录
        log_stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # 保存最佳模型
        if test_acc > max_accuracy:
            print(f"    New best model accuracy: {test_acc:.2f}%. Saving...")
            max_accuracy = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

    print("\n==> Training Finished!")
    print(f"Best model accuracy: {max_accuracy:.2f}%")
    print(f"Logs and checkpoints saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ResNet-Hybrid training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
