import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial
import numpy as np


def parse(x, n):
    if isinstance(x, collections.abc.Iterable):
        if len(x) == 1:
            return list(repeat(x[0], n))
        elif len(x) == n:
            return x
        else:
            raise ValueError('length of x should be 1 or n')
    else:
        return list(repeat(x, n))

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power and n < len(power_labels) -1 :
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

class Attention(nn.Module):
    def __init__(self, in_planes, reduction, num_static_cell, num_local_mixture,num_fourier_basis, norm_layer=nn.LayerNorm,
                 cell_num_ratio=1.0, nonlocal_basis_ratio=1.0, start_cell_idx=None):
        super(Attention, self).__init__()
        hidden_planes = max(int(in_planes * reduction), 16)
        self.kw_planes_per_mixture = num_static_cell + 1
        self.num_local_mixture = num_local_mixture
        self.num_fourier_basis = num_fourier_basis
        self.fourier_enabled = num_fourier_basis > 0
        self.spatial_enabled = num_static_cell > 0
        self.kw_planes = self.kw_planes_per_mixture * num_local_mixture

        self.num_local_cell = int(cell_num_ratio * num_local_mixture)
        self.num_nonlocal_cell = num_static_cell - self.num_local_cell
        self.start_cell_idx = start_cell_idx

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_planes, hidden_planes, bias=(norm_layer is not nn.BatchNorm1d))
        self.norm1 = norm_layer(hidden_planes)
        self.act1 = nn.ReLU(inplace=True)

        if nonlocal_basis_ratio >= 1.0:
            self.map_to_cell = nn.Identity()
            self.fc2 = nn.Linear(hidden_planes, self.kw_planes, bias=True)
        else:
            self.map_to_cell = self.map_to_cell_basis
            self.num_basis = max(int(self.num_nonlocal_cell * nonlocal_basis_ratio), 16)
            self.fc2 = nn.Linear(hidden_planes, (self.num_local_cell + self.num_basis + 1) * num_local_mixture, bias=False)
            self.fc3 = nn.Linear(self.num_basis, self.num_nonlocal_cell, bias=False)
            self.basis_bias = nn.Parameter(torch.zeros([self.kw_planes]), requires_grad=True).float()

        self.temp_bias = torch.zeros([self.kw_planes], requires_grad=False).float()
        self.temp_value = 0
        self._initialize_weights()
        if self.fourier_enabled:
            
            self.fourier_head = nn.Linear(hidden_planes, self.num_fourier_basis * in_planes, bias=True)
        if self.spatial_enabled:
            
            self.spatial_head = nn.Linear(hidden_planes, self.kw_planes, bias=True)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temp_value):
        self.temp_value = temp_value

    def init_temperature(self, start_cell_idx, num_cell_per_mixture):
        if num_cell_per_mixture >= 1.0:
            num_cell_per_mixture = int(num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                self.temp_bias[assigned_kernel_idx] = 1
                start_cell_idx += num_cell_per_mixture
            return start_cell_idx
        else:
            num_mixture_per_cell = int(1.0 / num_cell_per_mixture)
            for idx in range(self.num_local_mixture):
                if idx % num_mixture_per_cell == (idx // num_mixture_per_cell) % num_mixture_per_cell:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + start_cell_idx)
                    self.temp_bias[assigned_kernel_idx] = 1
                    start_cell_idx += 1
                else:
                    assigned_kernel_idx = int(idx * self.kw_planes_per_mixture + self.kw_planes_per_mixture - 1)
                    self.temp_bias[assigned_kernel_idx] = 1
            return start_cell_idx

    def map_to_cell_basis(self, x):
        x = x.reshape([-1, self.num_local_cell + self.num_basis + 1])
        x_local, x_nonlocal, x_zero = x[:, :self.num_local_cell], x[:, self.num_local_cell:-1], x[:, -1:]
        x_nonlocal = self.fc3(x_nonlocal)
        x = torch.cat([x_nonlocal[:, :self.start_cell_idx], x_local, x_nonlocal[:, self.start_cell_idx:], x_zero], dim=1)
        x = x.reshape(-1, self.kw_planes) + self.basis_bias.reshape(1, -1)
        return x

    def forward(self, x):
        x = self.avgpool(x.reshape(*x.shape[:2], -1)).squeeze(dim=-1)
        x = self.act1(self.norm1(self.fc1(x)))
        kw_attention_fourier = None
        if self.fourier_enabled:
            kw_attention_fourier = self.fourier_head(x)
        if self.spatial_enabled:
            kw_attention_spatial = self.spatial_head(x)
        x = self.map_to_cell(kw_attention_spatial).reshape(-1, self.kw_planes_per_mixture)
        x = x / (torch.sum(torch.abs(x), dim=1).view(-1, 1) + 1e-3) 
        x = (1.0 - self.temp_value) * x.reshape(-1, self.kw_planes) \
            + self.temp_value * self.temp_bias.to(x.device).view(1, -1)
        return x.reshape(-1, self.kw_planes_per_mixture)[:, :-1],kw_attention_fourier

def legendre_basis(H, W, k, device):
    """
    生成勒让德多项式基的傅里叶变换结果。
    
    Args:
        H (int): 特征图高度
        W (int): 特征图宽度
        k (int): 使用的多项式阶数 (基的数量)
        device: a torch device
    
    Returns:
        torch.Tensor: 预计算好的频域基, shape [k, H, W//2 + 1]
    """
    basis_list = []
    # 创建归一化的坐标网格 [-1, 1]
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    
    # 使用 numpy 的勒让德多项式函数来计算
    # i, j 是多项式的阶数
    for i in range(k):
        # 简单起见，我们只用一维的基 P_i(x) * P_i(y)，可以扩展到 P_i(x) * P_j(y)
        p_i_y = torch.from_numpy(np.polynomial.legendre.legval(y.cpu().numpy(), [0]*i + [1])).to(device).float()
        p_i_x = torch.from_numpy(np.polynomial.legendre.legval(x.cpu().numpy(), [0]*i + [1])).to(device).float()
        
        # 通过外积创建2D基
        basis_2d = torch.outer(p_i_y, p_i_x) # Shape: [H, W]
        
        # FFT 并存储
        basis_ft = torch.fft.rfft2(basis_2d, s=(H, W), norm='ortho')
        basis_list.append(basis_ft)
        
    return torch.stack(basis_list, dim=0)

class KWconvNd(nn.Module):
    dimension = None
    permute = None
    func_conv = None

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, warehouse_id=None, warehouse_manager=None):
        super(KWconvNd, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = parse(kernel_size, self.dimension)
        self.stride = parse(stride, self.dimension)
        self.padding = parse(padding, self.dimension)
        self.dilation = parse(dilation, self.dimension)
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros([self.out_planes]), requires_grad=True).float() if bias else None
        self.warehouse_id = warehouse_id
        self.warehouse_manager = [warehouse_manager]  # avoid repeat registration for warehouse manager
        self.fourier_basis_created = False
        self.num_fourier_basis = 1 
        self.fourier_coefficient_generator = None
        self.pointwise_mixer = nn.Conv2d(
            self.in_planes, self.out_planes, kernel_size=1, 
            stride=1, padding=0, groups=self.groups, bias=False
        )
    def init_attention(self, cell, num_total_attention_targets, num_fourier_basis, start_cell_idx, reduction, 
                         cell_num_ratio, norm_layer, nonlocal_basis_ratio=1.0):
        
        self.cell_shape = cell.shape 
        self.num_fourier_basis = num_fourier_basis
        self.num_spatial_cells = self.cell_shape[0] 
        
        # 验证总数是否匹配
        assert self.num_spatial_cells + self.num_fourier_basis == num_total_attention_targets

        self.groups_out_channel = self.out_planes // self.cell_shape[1]
        self.groups_in_channel = self.in_planes // self.cell_shape[2] // self.groups
        
        # --- FIX: 补回缺失的 self.groups_spatial 计算 ---
        self.groups_spatial = 1
        for idx in range(len(self.kernel_size)):
            self.groups_spatial = self.groups_spatial * self.kernel_size[idx] // self.cell_shape[3 + idx]
        # --- END FIX ---
        
        num_local_mixture = self.groups_out_channel * self.groups_in_channel * self.groups_spatial
        
        # 关键: Attention模块需要知道总的细胞数量，以便生成足够长的注意力向量
        self.attention = Attention(
            in_planes=self.in_planes, 
            reduction=reduction,
            # 空间路径参数
            num_static_cell=self.num_spatial_cells,
            num_local_mixture=num_local_mixture,
            # 频域路径参数
            num_fourier_basis=self.num_fourier_basis,
            # 其他
            norm_layer=norm_layer,
            nonlocal_basis_ratio=nonlocal_basis_ratio,
            start_cell_idx=start_cell_idx
        )
        
        
        if self.num_fourier_basis > 0:
            # 输入维度是 num_local_mixture * num_fourier_basis
            # 这是 kw_attention_fourier 的展平维度
            generator_in_dim = num_local_mixture * self.num_fourier_basis
            self.fourier_path_gamma = nn.Parameter(torch.zeros(1, self.out_planes, 1, 1))
            # 输出维度是我们需要的系数总数
            # 我们为每个 (输出通道, 输入通道/组) 生成 k 个系数
            generator_out_dim = self.out_planes * (self.in_planes // self.groups) * self.num_fourier_basis
            self.pointwise_mixer_bn = nn.BatchNorm2d(self.out_planes)
            # 创建一个简单的线性层作为系数生成器
            self.fourier_coefficient_generator = nn.Sequential(
                nn.Linear(generator_in_dim, generator_out_dim),
                # 可以选择性地添加激活函数或归一化层
                # nn.LayerNorm(generator_out_dim) 
            )
        return self.attention.init_temperature(start_cell_idx, cell_num_ratio)
    def _create_local_fourier_basis_if_needed(self, patch_size, device):
        """
        一个惰性初始化函数，在第一次前向传播时创建局部的、共享的频域基。
        """
        # 只有在需要且尚未创建时才执行
        if self.num_fourier_basis > 0 and not self.fourier_basis_created:
            # 调用辅助函数创建基，尺寸为 patch_size x patch_size
            basis = legendre_basis(patch_size, patch_size, self.num_fourier_basis, device=device)
            # 使用 register_buffer 将其注册为模块的一部分。
            # 这样它会随 .to(device) 移动，并且不会被当成可训练参数。
            self.register_buffer('local_fourier_basis', basis)
            self.fourier_basis_created = True

    def forward(self, x):
        kw_attention ,kw_attention_fourier = self.attention(x)
        # 前 num_fourier_cells 个权重给频域细胞
        patch_size = 10
        self._create_local_fourier_basis_if_needed(patch_size, x.device)
        batch_size ,C_in,H,W= x.shape
        output = 0.0
        device = x.device
        
        # --- 路径A: 空间域卷积 ---
        if self.num_spatial_cells > 0:
            spatial_cells = self.warehouse_manager[0].take_cell(self.warehouse_id).reshape(self.num_spatial_cells, -1)
            aggregate_weight_spatial = torch.mm(kw_attention, spatial_cells)
            
            # (这部分reshape和permute逻辑与原始代码完全相同)
            aggregate_weight_spatial = aggregate_weight_spatial.reshape([batch_size, self.groups_spatial, self.groups_out_channel,
                                                                         self.groups_in_channel, *self.cell_shape[1:]])
            aggregate_weight_spatial = aggregate_weight_spatial.permute(*self.permute)
            aggregate_weight_spatial = aggregate_weight_spatial.reshape(-1, self.in_planes // self.groups, *self.kernel_size)
            
            x_spatial = x.reshape(1, -1, *x.shape[2:])
            output_spatial = self.func_conv(x_spatial, weight=aggregate_weight_spatial, bias=None, stride=self.stride, padding=self.padding,
                                            dilation=self.dilation, groups=self.groups * batch_size)
            output_spatial = output_spatial.view(batch_size, self.out_planes, *output_spatial.shape[2:])
            output = output + output_spatial

        output_fourier = None

        if self.num_fourier_basis > 0 and kw_attention_fourier is not None:
            B, C_in, H, W = x.shape
            
            # --- 1. 向 Manager 请求一个与当前 H, W 完全匹配的基 ---
            manager = self.warehouse_manager[0]
            fourier_basis = manager.get_or_create_fourier_basis(self.warehouse_id, H, W, x.device)
            
            if fourier_basis is not None:
                # 2. 全局 FFT
                x_ft = torch.fft.rfft2(x, s=(H, W), norm='ortho')

                # 3. 动态生成 Depthwise 系数
                coefficients_real = kw_attention_fourier.reshape(B, self.in_planes, self.num_fourier_basis)
                coefficients = torch.complex(coefficients_real, torch.zeros_like(coefficients_real))
                
                # 4. 合成 Depthwise 全局频域核
                # 此时 fourier_basis 的尺寸与 x_ft 保证匹配！
                kernel_ft_dw = torch.einsum("bik,khw->bihw", coefficients, fourier_basis)

                # 5. 执行 Depthwise 滤波
                output_ft_dw = x_ft * kernel_ft_dw
                is_strided = any(s > 1 for s in self.stride)
                if is_strided:
                    H_out = H // self.stride[0]
                    W_out = W // self.stride[1]
                    W_out_f = W_out // 2 + 1
                    # 只保留与目标输出尺寸对应的低频部分
                    output_ft_dw = output_ft_dw[..., :H_out, :W_out_f]
                    # IFFT的目标尺寸也变为小尺寸
                    target_s = (H_out, W_out)
                else:
                    target_s = (H, W)
                # 6. 逆FFT
                output_fourier_dw = torch.fft.irfft2(output_ft_dw, s=target_s, norm='ortho')
                output_fourier_mixed = self.pointwise_mixer_bn(self.pointwise_mixer(output_fourier_dw))
                
                # 8. 处理步长
                output_fourier = output_fourier_mixed

        # --- 最终融合 ---
        if output_spatial is not None and output_fourier is not None:
            assert output_spatial.shape == output_fourier.shape, \
                   f"Shape mismatch: spatial {output_spatial.shape} vs fourier {output_fourier.shape}"
            # 使用可学习的 gamma 进行稳定融合
            # output = output_spatial + self.fourier_path_gamma * output_fourier
        elif output_spatial is not None:
            output = output_spatial
        elif output_fourier is not None:
            output = output_fourier
        else:
            raise ValueError("Both spatial and fourier paths are disabled, which is not expected.")

        # 添加偏置
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, *([1]*self.dimension))
        
        return output

class KWConv1d(KWconvNd):
    dimension = 1
    permute = (0, 2, 4, 3, 5, 1, 6)
    func_conv = F.conv1d


class KWConv2d(KWconvNd):
    dimension = 2
    permute = (0, 2, 4, 3, 5, 1, 6, 7)
    func_conv = F.conv2d


class KWConv3d(KWconvNd):
    dimension = 3
    permute = (0, 2, 4, 3, 5, 1, 6, 7, 8)
    func_conv = F.conv3d


class KWLinear(nn.Module):
    dimension = 1

    def __init__(self, *args, **kwargs):
        super(KWLinear, self).__init__()
        self.conv = KWConv1d(*args, **kwargs)

    def forward(self, x):
        shape = x.shape
        x = self.conv(x.reshape(shape[0], -1, shape[-1]).transpose(1, 2))
        x = x.transpose(1, 2).reshape(*shape[:-1], -1)
        return x


class Warehouse_Manager(nn.Module):
    def __init__(self, reduction=0.0625, cell_num_ratio=1, cell_inplane_ratio=1,
                 cell_outplane_ratio=1, sharing_range=(), nonlocal_basis_ratio=1,
                 norm_layer=nn.BatchNorm1d, spatial_partition=True,num_fourier_basis=9,fourier_basis_ratio=1.0):
        """
        Create a Kernel Warehouse manager for a network.
        Args:
            reduction (float or tuple): reduction ratio for hidden plane
            cell_num_ratio (float or tuple): number of kernel cells in warehouse / number of kernel cells divided
                        from convolutional layers, set cell_num_ratio >= max(cell_inplane_ratio, cell_outplane_ratio)
                        for applying temperature initialization strategy properly
            cell_inplane_ratio (float or tuple): input channels of kernel cells / the greatest common divisor for
                        input channels of convolutional layers
            cell_outplane_ratio (float or tuple): input channels of kernel cells / the greatest common divisor for
                        output channels of convolutional layers
            sharing_range (tuple): range of warehouse sharing.
                        For example, if the input is ["layer", "conv"], the convolutional layer "stageA_layerB_convC"
                        will be assigned to the warehouse "stageA_layer_conv"
            nonlocal_basis_ratio (float or tuple): reduction ratio for mapping kernel cells belongs to other layers
                        into fewer kernel cells in the attention module of a layer to reduce parameters, enabled if
                        nonlocal_basis_ratio < 1.
            spatial_partition (bool or tuple): If ``True``, splits kernels into cells along spatial dimension.
        """
        super(Warehouse_Manager, self).__init__()
        self.sharing_range = sharing_range
        self.warehouse_list = {}
        self.reduction = reduction
        self.spatial_partition = spatial_partition
        self.cell_num_ratio = cell_num_ratio
        self.cell_outplane_ratio = cell_outplane_ratio
        self.cell_inplane_ratio = cell_inplane_ratio
        self.norm_layer = norm_layer
        self.nonlocal_basis_ratio = nonlocal_basis_ratio
        self.num_fourier_basis = num_fourier_basis
        self.fourier_basis_ratio = fourier_basis_ratio
        self.num_fourier_basis_per_warehouse = []
    def fuse_warehouse_name(self, warehouse_name):
        fused_names = []
        for sub_name in warehouse_name.split('_'):
            match_name = sub_name
            for sharing_name in self.sharing_range:
                if str.startswith(match_name, sharing_name):
                    match_name = sharing_name
            fused_names.append(match_name)
        fused_names = '_'.join(fused_names)
        return fused_names
    def get_or_create_fourier_basis(self, warehouse_id, H, W, device):
        # 使用存储好的数量
        num_basis = self.num_fourier_basis_per_warehouse[warehouse_id]
        if num_basis <= 0: return None
        buffer_name = f'fourier_basis_w{warehouse_id}_{H}x{W}'
        
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)
        else:
            print(f"INFO: WM creating and registering Buffer '{buffer_name}' with {num_basis} bases on device {device}")
            basis = legendre_basis(H, W, num_basis, device=device)
            self.register_buffer(buffer_name, basis)
            return basis
    def reserve(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                bias=True, warehouse_name='default', enabled=True, layer_type='conv2d'):
        """
        Create a dynamic convolution layer without convolutional weights and record its information.
        Args:
            warehouse_name (str): the warehouse name of current layer
            enabled (bool): If ``False``, return a vanilla convolutional layer defined in pytorch.
            layer_type (str): 'conv1d', 'conv2d', 'conv3d' or 'linear'
        """
        kw_mapping = {'conv1d': KWConv1d, 'conv2d': KWConv2d, 'conv3d': KWConv3d, 'linear': KWLinear}
        org_mapping = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'linear': nn.Linear}

        if not enabled:
            layer_type = org_mapping[layer_type]
            if layer_type is nn.Linear:
                return layer_type(in_planes, out_planes, bias=bias)
            else:
                return layer_type(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
        else:
            layer_type = kw_mapping[layer_type]
            warehouse_name = self.fuse_warehouse_name(warehouse_name)
            weight_shape = [out_planes, in_planes // groups, *parse(kernel_size, layer_type.dimension)]

            if warehouse_name not in self.warehouse_list.keys():
                self.warehouse_list[warehouse_name] = []
            self.warehouse_list[warehouse_name].append(weight_shape)

            return layer_type(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias,
                              warehouse_id=int(list(self.warehouse_list.keys()).index(warehouse_name)),
                              warehouse_manager=self)

    def store(self):
        warehouse_names = list(self.warehouse_list.keys())
        self.reduction = parse(self.reduction, len(warehouse_names))
        self.spatial_partition = parse(self.spatial_partition, len(warehouse_names))
        self.cell_num_ratio = parse(self.cell_num_ratio, len(warehouse_names))
        self.cell_outplane_ratio = parse(self.cell_outplane_ratio, len(warehouse_names))
        self.cell_inplane_ratio = parse(self.cell_inplane_ratio, len(warehouse_names))
        self.num_fourier_basis = parse(self.num_fourier_basis, len(warehouse_names))
        self.fourier_basis_ratio = parse(self.fourier_basis_ratio, len(self.warehouse_names))
        self.weights = nn.ParameterList()
        self.num_fourier_basis_per_warehouse = []

        for idx, warehouse_name in enumerate(self.warehouse_list.keys()):
            warehouse = self.warehouse_list[warehouse_name]
            dimension = len(warehouse[0]) - 2

            # Calculate the greatest common divisors
            out_plane_gcd, in_plane_gcd, kernel_size = warehouse[0][0], warehouse[0][1], warehouse[0][2:]
            for layer in warehouse:
                out_plane_gcd = math.gcd(out_plane_gcd, layer[0])
                in_plane_gcd = math.gcd(in_plane_gcd, layer[1])
                if not self.spatial_partition[idx]:
                    assert kernel_size == layer[2:]

            cell_in_plane = max(int(in_plane_gcd * self.cell_inplane_ratio[idx]), 1)
            cell_out_plane = max(int(out_plane_gcd * self.cell_outplane_ratio[idx]), 1)
            cell_kernel_size = parse(1, dimension) if self.spatial_partition[idx] else kernel_size

            # Calculate number of total mixtures to calculate for each stage
            num_total_mixtures = 0
            for layer in warehouse:
                groups_channel = int(layer[0] // cell_out_plane * layer[1] // cell_in_plane)
                groups_spatial = 1

                for d in range(dimension):
                    groups_spatial = int(groups_spatial * layer[2 + d] // cell_kernel_size[d])

                num_layer_mixtures = groups_spatial * groups_channel
                num_total_mixtures += num_layer_mixtures
            num_spatial_cells = max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1)
            self.weights.append(nn.Parameter(torch.randn(
                max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
                cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True))
            ratio = self.fourier_basis_ratio[idx]
            if ratio > 0:
                num_basis = max(int(num_spatial_cells * ratio), 1)
                self.num_fourier_basis_per_warehouse.append(num_basis)
                print(f"INFO: WM configured warehouse {idx} to have {num_basis} Fourier basis (ratio={ratio})")
            else:
                self.num_fourier_basis_per_warehouse.append(0)
    def allocate(self, network, _init_weights=partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')):
        num_warehouse = len(self.weights)
        parsed_num_fourier_basis = parse(self.num_fourier_basis, num_warehouse)
        end_idxs = [0] * num_warehouse

        for layer in network.modules():
            if isinstance(layer, KWconvNd):
                warehouse_idx = layer.warehouse_id
                start_cell_idx = end_idxs[warehouse_idx]

                # 空间细胞的数量由仓库决定
                num_spatial_cells = self.weights[warehouse_idx].shape[0]
                # 频域基的数量由配置决定
                num_fourier_basis = 9
                
                # 总的“注意力目标”数量
                total_attention_targets = num_spatial_cells + num_fourier_basis

                # 将所有必要的配置传递给 init_attention 方法
                end_cell_idx = layer.init_attention(
                    cell=self.weights[warehouse_idx],
                    num_total_attention_targets=total_attention_targets,
                    num_fourier_basis=num_fourier_basis,
                    start_cell_idx=start_cell_idx,
                    reduction=self.reduction[warehouse_idx],
                    cell_num_ratio=self.cell_num_ratio[warehouse_idx],
                    norm_layer=self.norm_layer,
                    nonlocal_basis_ratio=self.nonlocal_basis_ratio
                )
                
                # FIX: 只初始化空间权重，因为没有 fourier_weights 了
                _init_weights(self.weights[warehouse_idx][start_cell_idx:end_cell_idx].view(
                    -1, *self.weights[warehouse_idx].shape[2:]))

                end_idxs[warehouse_idx] = end_cell_idx

        for warehouse_idx in range(len(end_idxs)):
            assert end_idxs[warehouse_idx] == self.weights[warehouse_idx].shape[0]

    def take_cell(self, warehouse_idx):
        return self.weights[warehouse_idx]
    def take_fourier_basis(self, warehouse_idx):
        return self.fourier_bases[warehouse_idx]
