import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial


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


class Attention(nn.Module):
    def __init__(self, in_planes, reduction, num_static_cell, num_local_mixture, norm_layer=nn.BatchNorm1d,
                 cell_num_ratio=1.0, nonlocal_basis_ratio=1.0, start_cell_idx=None):
        super(Attention, self).__init__()
        hidden_planes = max(int(in_planes * reduction), 16)
        self.kw_planes_per_mixture = num_static_cell + 1
        self.num_local_mixture = num_local_mixture
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
        x = self.map_to_cell(self.fc2(x)).reshape(-1, self.kw_planes_per_mixture)
        x = x / (torch.sum(torch.abs(x), dim=1).view(-1, 1) + 1e-3)
        x = (1.0 - self.temp_value) * x.reshape(-1, self.kw_planes) \
            + self.temp_value * self.temp_bias.to(x.device).view(1, -1)
        return x.reshape(-1, self.kw_planes_per_mixture)[:, :-1]

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
        self.num_fourier_basis = 0
        self.fourier_basis_created = False
    def init_attention(self, cell, num_total_attention_targets, num_fourier_basis, start_cell_idx, reduction, 
                         cell_num_ratio, norm_layer, nonlocal_basis_ratio=1.0):
        
        self.cell_shape = cell.shape # 形状信息来自空间细胞仓库
        self.num_fourier_basis = num_fourier_basis
        self.num_spatial_cells = self.cell_shape[0] # 空间细胞数就是传入的cell仓库的大小
        
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
        self.attention = Attention(self.in_planes, reduction, num_total_attention_targets, num_local_mixture,
                                   norm_layer=norm_layer, nonlocal_basis_ratio=nonlocal_basis_ratio,
                                   cell_num_ratio=cell_num_ratio, start_cell_idx=start_cell_idx)
        
        # 这个返回值现在只对原始的空间细胞分配策略有意义
        return self.attention.init_temperature(start_cell_idx, cell_num_ratio)
    def _create_fourier_basis(self, H, W, device):
        """
        在第一次前向传播时，根据实际输入尺寸创建频域基。
        """
        if self.num_fourier_basis > 0 and not self.fourier_basis_created:
            basis = legendre_basis(H, W, self.num_fourier_basis, device=device)
            # 注册为 buffer，使其成为模块的一部分并随模块移动设备
            self.register_buffer('fourier_basis', basis)
            self.fourier_basis_created = True

    def forward(self, x):
        kw_attention = self.attention(x)
        # 前 num_fourier_cells 个权重给频域细胞
        kw_attention_fourier = kw_attention[:, :self.num_fourier_basis]
        # 后面 num_spatial_cells 个权重给空间细胞
        kw_attention_spatial = kw_attention[:, self.num_fourier_basis:]
        batch_size = x.shape[0]
        output = 0.0
        device = x.device
        # --- 路径A: 空间域卷积 ---
        if self.num_spatial_cells > 0:
            spatial_cells = self.warehouse_manager[0].take_cell(self.warehouse_id).reshape(self.num_spatial_cells, -1)
            aggregate_weight_spatial = torch.mm(kw_attention_spatial, spatial_cells)
            
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

        # --- 路径B: 频率域卷积 ---
         # --- 路径B: 多项式基展开频域卷积 (全新实现) ---
        if self.num_fourier_basis > 0:
            H, W = x.shape[-2:]
            self._create_fourier_basis(H, W, x.device)
            # a. 对输入 x 进行 FFT
            x_ft = torch.fft.rfft2(x, s=(H, W), norm='ortho')
            
            # b. 获取预计算的频域基
            # fourier_basis 的形状是 [k, H, W//2+1]
            fourier_basis = self.fourier_basis
            
            # c. 将动态注意力权重塑造成组合系数
            # kw_attention_fourier 的形状是 [B*G, k]
            # 我们需要将其 reshape 成 [B, C_out, C_in_g, k] 的形式
            # 这是一个设计选择，最简单的方式是让注意力直接为每个 (C_out, C_in) 生成 k 个系数
            # 这需要 Attention 模块的输出维度匹配 B * C_out * C_in_g * k
            # 为了演示，我们做一个简化，假设注意力权重在通道间共享
            # kw_attention_fourier: [B*G, k]
            # G = groups_out_channel * groups_in_channel * groups_spatial
            # Reshape a_f to [B, C_out, C_in_g, k] (需要你的 Attention 支持)
            # 这里我们做一个简化的reshape，实际应用中可能需要更复杂的映射
            coefficients = kw_attention_fourier.reshape(
                batch_size, 
                self.groups_out_channel * self.groups_in_channel * self.groups_spatial, 
                self.num_fourier_basis
            ) 
            # 简化假设：输出通道数等于输入通道数，且 groups=1, G=1
            # 这样 coefficients 的形状是 [B, C_out, k] (假设在输入通道间共享)
            # 这是一个非常强的假设，仅为演示einsum
            if self.out_planes != self.in_planes or self.groups != 1:
                # 在实际应用中，你需要确保 Attention 模块能输出正确数量的系数
                # print("Warning: A simplified coefficient mapping is used for demonstration.")
                # 这里我们假设输出通道和输入通道相同，且groups=1
                # 并且将注意力权重广播到所有通道
                coefficients = kw_attention_fourier.reshape(batch_size, 1, self.num_fourier_basis) # [B, 1, k]

            # d. 在频域中通过线性组合“合成”卷积核
            # (B, 1, k) x (k, H, W_f) -> (B, H, W_f) (在通道间共享核)
            # 然后广播到: (B, C, H, W_f)
            # kernel_ft_shared = torch.einsum("bik,khw->bihw", coefficients, fourier_basis)
            # output_ft = kernel_ft_shared * x_ft # 逐元素乘法

            # 更通用、更强大的 einsum (假设 Attention 能生成 [B, C_out, C_in, k] 的系数)
            # coefficients = ... # [B, C_out, C_in, k]
            # fourier_basis: [k, H, W_f]
            # kernel_ft = torch.einsum("boik,khw->boihw", coefficients, fourier_basis)
            
            # --- 为了让代码能跑起来，我们做一个合理的简化 ---
            # 让 Attention 为每个输出通道生成k个系数，并在输入通道间共享
            # 这需要 Attention 的输出维度是 hidden -> num_local_mixture * num_fourier_basis
            # kw_attention_fourier: [B * G, k], 其中 G = C_out/cell_c_out * ...
            # 我们可以reshape成 [B, C_out, k] (假设 G = C_out)
            coefficients = kw_attention_fourier.reshape(batch_size, self.out_planes, self.num_fourier_basis) # [B, o, k]
            
            # kernel_ft: (B, o, k) x (k, H, W_f) -> (B, o, H, W_f)
            kernel_ft = torch.einsum('bok,khw->bohw', coefficients, fourier_basis)
            
            # e. 频域乘法 (卷积)
            # x_ft: (B, i, H, W_f)
            # kernel_ft: (B, o, H, W_f)
            # 我们希望在输入通道 i 上求和
            output_ft = torch.einsum("bihw,bohw->bohw", x_ft, kernel_ft)

            # f. 逆变换
            output_fourier = torch.fft.irfft2(output_ft, s=(H, W), norm='ortho')
            
            output = output + output_fourier
        # --- 4. 添加偏置项 ---
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
                 norm_layer=nn.BatchNorm1d, spatial_partition=True,num_fourier_basis=0):
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
        self.weights = nn.ParameterList()
        self.fourier_bases = []

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

            self.weights.append(nn.Parameter(torch.randn(
                max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
                cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True))

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
                num_fourier_basis = parsed_num_fourier_basis[warehouse_idx]
                
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

