import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial

from triton.language import dtype

a = 0.2

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

    def init_attention(self, cell, start_cell_idx, reduction, cell_num_ratio, norm_layer, nonlocal_basis_ratio=1.0,num_F=0):
        self.cell_shape = cell.shape # [M, C_{out}, C_{in}, D, H, W]
        self.groups_out_channel = self.out_planes // self.cell_shape[1]
        self.groups_in_channel = self.in_planes // self.cell_shape[1] // self.groups
        self.groups_spatial = 1

        self.norm_cell_num = self.cell_shape[0] - num_F
        for idx in range(len(self.kernel_size)):
            self.groups_spatial = self.groups_spatial * self.kernel_size[idx] // self.cell_shape[3 + idx]
        num_local_mixture = self.groups_out_channel * self.groups_in_channel * self.groups_spatial
        self.attention = Attention(self.in_planes, reduction, self.norm_cell_num, num_local_mixture,
                                   norm_layer=norm_layer, nonlocal_basis_ratio=nonlocal_basis_ratio,
                                   cell_num_ratio=cell_num_ratio, start_cell_idx=start_cell_idx)
        return self.attention.init_temperature(start_cell_idx, cell_num_ratio)
    def forward(self, x):
        self.num_F = self.warehouse_manager[0].take_numF(self.warehouse_id)
        kw_attention = self.attention(x)
        Fkw_attention = kw_attention[:,-self.num_F:]
        kw_attention = kw_attention[:,:-self.num_F]
        batch_size = x.shape[0]
        x = x.reshape(1, -1, *x.shape[2:])
        weight = self.warehouse_manager[0].take_cell(self.warehouse_id).reshape(self.cell_shape[0], -1)
        # weight = weight[:(self.cell_shape[0] - self.num_F),:]
        # F_weight = weight[-self.num_F:,:]
        # weight = self.Fourier(self.cell_shape[0],self.cell_shape[1] * self.cell_shape[2]).cuda()
        aggregate_weight = torch.mm(kw_attention, weight)

        aggregate_weight = aggregate_weight.reshape([batch_size, self.groups_spatial, self.groups_out_channel,
                                                     self.groups_in_channel, *self.cell_shape[1:]])
        aggregate_weight = aggregate_weight.permute(*self.permute)
        aggregate_weight = aggregate_weight.reshape(-1, self.in_planes // self.groups, *self.kernel_size)
        output = self.func_conv(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, *output.shape[2:])
        x_fft = torch.fft.fft2(x)  # 对输入x进行傅里叶变换
        weight_fft = torch.fft.fft2(Fkw_attention, s=x.shape[2:])
        output_fft = x_fft * weight_fft
        output_fft = torch.fft.ifft2(output_fft).real  # 进行逆傅里叶变换，得到空间域的输出
        # F_weight = torch.fft.fft2(F_weight)
        # Fkw_attention = torch.fft.fft2(Fkw_attention)
        # F_aggregate_weight = torch.mm(Fkw_attention, F_weight)
        #
        # F_aggregate_weight = F_aggregate_weight.reshape([batch_size, self.groups_spatial, self.groups_out_channel,
        #                                              self.groups_in_channel, *self.cell_shape[1:]])
        # F_aggregate_weight = F_aggregate_weight.permute(*self.permute)
        # F_aggregate_weight = F_aggregate_weight.reshape(-1, self.in_planes // self.groups, *self.kernel_size)
        # F_x = torch.fft.fft2(x.to(torch.float32))
        # F_output = self.func_conv(F_x, weight=F_aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
        #                         dilation=self.dilation, groups=self.groups * batch_size)
        # F_output = torch.fft.ifft2(F_output).real
        # F_output = F_output.view(batch_size, self.out_planes, *output.shape[2:])
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, *([1]*self.dimension))
            # F_output = F_output + self.bias.reshape(1, -1, *([1] * self.dimension))
        output = output +output_fft
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
                 norm_layer=nn.BatchNorm1d, spatial_partition=True):
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
        self.weights = nn.ParameterList()
        self.select =torch.rand(())
        self.num_F = []

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
            num_fourier = math.ceil(num_total_mixtures * a)
            num_norm = num_total_mixtures - num_fourier
            E = nn.Parameter(torch.randn(num_fourier * self.cell_num_ratio[idx],cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True)
            norm_conv_weight = nn.Parameter(torch.randn(max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
            cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True)
            # E = torch.cat((norm_conv_weight,E),dim=0)
            # frequency_normavg = torch.mean(E_frequency[:num_norm],dim=0).unsqueeze(0)
            # with torch.no_grad():
            #     avg_frequency = (frequency_normavg + E_frequency[num_fourier:] /2)
            # E = torch.cat((E[:num_norm],E_frequency[torch.randint(num_fourier,num_total_mixtures,(int(num_fourier/2),))],
            #                avg_frequency[torch.randint(num_fourier,(int(num_fourier/2),))]),dim=0)

            all_weight = torch.cat(
                (norm_conv_weight, E),dim=0)
            # self.weights.append(nn.Parameter(torch.randn(
            #     max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
            #     cell_out_plane, cell_in_plane, *cell_kernel_siz), requires_grad=True))
            self.weights.append(all_weight)
            self.num_F.append(num_fourier)
    def allocate(self, network, _init_weights=partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')):
        num_warehouse = len(self.weights)
        end_idxs = [0] * num_warehouse

        for layer in network.modules():
            if isinstance(layer, KWconvNd):
                warehouse_idx = layer.warehouse_id
                start_cell_idx = end_idxs[warehouse_idx]
                end_cell_idx = layer.init_attention(self.weights[warehouse_idx],
                                                    start_cell_idx,
                                                    self.reduction[warehouse_idx],
                                                    self.cell_num_ratio[warehouse_idx],
                                                    norm_layer=self.norm_layer,
                                                    nonlocal_basis_ratio=self.nonlocal_basis_ratio,
                                                    num_F= self.num_F[warehouse_idx])
                _init_weights(self.weights[warehouse_idx][start_cell_idx:end_cell_idx].view(
                    -1, *self.weights[warehouse_idx].shape[2:]))
                end_idxs[warehouse_idx] = end_cell_idx

        for warehouse_idx in range(len(end_idxs)):
            print(end_idxs[warehouse_idx])
            print(self.weights[warehouse_idx].shape[0])
            # assert end_idxs[warehouse_idx] == self.weights[warehouse_idx].shape[0]

    def take_cell(self, warehouse_idx):
        return self.weights[warehouse_idx]
    def take_numF(self,warehouse_idx):
        return self.num_F[warehouse_idx]


