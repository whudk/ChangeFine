import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_,to_2tuple
import math
import einops

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,stride = 16):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels= in_dim * 2, out_channels=in_dim,kernel_size=1,stride = 1),
        #     ModuleHelper.BNReLU(in_dim),
        #     nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        # )
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//stride, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//stride, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    #     self._init_weights()
    # def _init_weights(self):
    #     for module in self.mlp:
    #         kaiming_init(module,mode='fan_in')
    #     kaiming_init(self.query_conv,mode='fan_in')
    #     kaiming_init(self.key_conv,mode='fan_in')
    #     kaiming_init(self.value_conv,mode='fan_in')
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        #x = self.mlp(x_)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class channelsAttention(nn.Module):
    def __init__(self, in_dim):
        super(channelsAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_dim//16, out_channels=in_dim, kernel_size=1),
        )
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        # batchsize,c,h,w = x.shape
        # x = x.permute(0,2,3,1).view(batchsize,-1,c)


        avg_out = self.mlp(self.avgpool(x))
        max_out = self.mlp(self.maxpool(x))
        out = avg_out + max_out
        attention = self.sigmod(out)

        return  attention * x
class spatialAttention(nn.Module):
    def __init__(self):
        super(spatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2,1,kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # batchsize,c,h,w = x.shape
        # x = x.permute(0,2,3,1).view(batchsize,-1,c)

        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat((avg_out,max_out),dim=1)
        out = self.conv1(out)
        attention = self.sigmod(out)
        return attention * x

class CAM_PAM(nn.Module):
    def __init__(self,dim):
        super(CAM_PAM, self).__init__()
        self.att = nn.Sequential(
            channelsAttention(dim),
            spatialAttention()
        )
    def forward(self,x):
        if len(x.shape) == 3:
            B, N , C= x.size()
            h = int(math.sqrt(N))
            x = einops.rearrange(x, 'b (h w) c  -> b c h w', h = h)
        x = x + self.att(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        if len(x.shape) == 4:
            _, _, H, W = x.size()
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        return x




class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = int(channels * expansion)
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        if  len(x.shape) == 4:
            _, _, H, W = x.size()
            x = einops.rearrange(x, 'b c h w -> b (h w) c')

        x = self.chunk(x)
        #x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class SCAM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=8, qk_scale=None, attn_drop=0., act_layer=nn.GELU, proj_drop=0., feats_exchange = 'CCMAT'):
        #inter_type:cmt,cat,ccmat
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        self.H = H
        self.W = W
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 预计算邻域索引（直接存储为张量）
        self.neighbor_indices = self._precompute_neighbors()
        self.neighbor_norm = nn.LayerNorm(embed_dim)

    def _precompute_neighbors(self):
        """
        预计算所有像素的邻域索引，并存储为张量格式
        """
        neighbors_list = []
        for idx in range(self.H * self.W):
            row, col = idx // self.W, idx % self.W
            neighbors = []
            for i in range(-self.window_size, self.window_size + 1):
                for j in range(-self.window_size, self.window_size + 1):
                    r, c = row + i, col + j
                    if 0 <= r < self.H and 0 <= c < self.W:
                        neighbors.append(r * self.W + c)
            neighbors_list.append(neighbors)

        # 转为 PyTorch Tensor，确保索引并行计算
        max_neighbors = max(len(n) for n in neighbors_list)  # 找到最大邻域数
        padded_neighbors = [n + [n[0]] * (max_neighbors - len(n)) for n in neighbors_list]  # 填充
        return torch.tensor(padded_neighbors, dtype=torch.long)  # [N, max_neighbors]

    def forward(self, x1, x2):
        """
        x1, x2: [B, N, D]  ->  B 批量大小, N = H*W, D 特征维度
        """
        B, N, D = x1.shape
        device = x1.device

        # 获取邻域索引（加速）
        neighbors_idx = self.neighbor_indices.to(device)  # [N, max_neighbors]

        # 直接索引所有邻域特征 (批量化)
        neighbor_feats = x2[:, neighbors_idx, :]  # [B, N, max_neighbors, D]

        # 获取中心特征
        center_feat = x2.unsqueeze(2)  # [B, N, 1, D]

        # 计算注意力权重 (批量方式)
        attn_weights = F.softmax(torch.einsum('bncd,bnkd->bnkc', center_feat, neighbor_feats) / (D ** 0.5),
                                 dim=-1)  # [B, N, max_neighbors, 1]

        # 加权融合
        x2_fused = (neighbor_feats * attn_weights).sum(dim=2)  # [B, N, D]
        x2_fused = self.neighbor_norm(x2_fused)
        # 标准 Cross-Attention (x1 as Query, x2_fused as Key, Value)
        attn_output, attn_map = self.attn(query=x2_fused, key=x1, value=x1)  # [B,N,D]   查询相似 attn_output

        return attn_output



class CCMAT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=8, qk_scale=None, attn_drop=0., act_layer=nn.GELU, proj_drop=0., feats_exchange = 'CCMAT'):
        #inter_type:cmt,cat,ccmat
        super().__init__()
        self.CCT = Cross_ChannelsAttention(dim,   num_heads = num_heads,qkv_bias=qkv_bias,window_size = window_size,qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.CMT = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,window_size=window_size, qk_scale=qk_scale, attn_drop=attn_drop,proj_drop=proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=act_layer, drop=proj_drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=act_layer, drop=proj_drop)
        self.feats_exchange = feats_exchange
        if self.feats_exchange not in ["CMT","CCT","CCMAT","NONE"]:
            raise Exception("interaction type {} is not supported".format(self.feats_exchange))
    def forward(self, x1,x2):
        if self.feats_exchange == "NONE":
            return  x1, x2
        if self.feats_exchange == "CMT":
            y1, y2, _, _ = self.CMT(self.norm(x1), self.norm(x2))
        elif self.feats_exchange == "CCT":
            y1, y2,_  = self.CCT(self.norm(x1), self.norm(x2))
        elif self.feats_exchange == "CCMAT":
            y1, y2, _, _ = self.CMT(self.norm(x1), self.norm(x2))

            #y3, y4, _, _ = self.CMT(self.norm(x2), self.norm(x1))
            #y1, y2,_  = self.CCT(self.norm1(y1), self.norm1(y2))
            y1, y2, _ = self.CCT(y1, y2)

        # x1 = x1 + self.drop_path(y1)
        # x1 = x1 + self.drop_path(self.mlp1(self.norm1(x1)))
        #
        # x2 = x2 + self.drop_path(y2)
        # x2 = x2 + self.drop_path(self.mlp2(self.norm2(x2)))

        x1 = x1 + self.drop_path(y1)
        x1 = x1 + self.drop_path(self.mlp1(self.norm2(x1)))

        x2 = x2 + self.drop_path(y2)
        x2 = x2 + self.drop_path(self.mlp1(self.norm2(x2)))

        return  x1, x2 ,_



class ChannelsAttention(nn.Module):
    def __init__(self, dim,   num_heads = 8,qkv_bias=False,window_size = 8,qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ChannelsAttention,self).__init__()

        window_size = to_2tuple(window_size)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)


        q = q * self.scale
        q = einops.rearrange(q, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)
        k = einops.rearrange(k, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)
        v = einops.rearrange(v, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)

        #v2 = self.v2(x2)
        #attn = torch.einsum('b h m n, b h m c -> b h m n',q,k)
        attn = q * k
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = attn * v#torch.einsum('b h m n, b h m c -> b h m c', attn, v)
        y = einops.rearrange(y, 'b h n c1 -> b n (h c1)')
        y = self.proj(y)
        y = self.proj_drop(y)

        x = y * self.parm + x

        return x, attn
class Cross_ChannelsAttention(nn.Module):
    def __init__(self, dim,   num_heads = 8,qkv_bias=False,window_size = 8,qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        window_size = to_2tuple(window_size)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N ,C = x1.size()
        #r1, r2 = H // self.window_size[0], W // self.window_size[1]






        q = self.q(x1)
        k = self.k(x2)
        v1 = self.v(x1)
        v2 = self.v(x2)

        q = q * self.scale
        q = einops.rearrange(q, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)
        k = einops.rearrange(k, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)
        v1 = einops.rearrange(v1, 'b n (h1 c1) -> b h1 n c1', h1 = self.num_heads)
        v2 = einops.rearrange(v2, 'b n (h1 c1) -> b h1 n c1', h1=self.num_heads)
        #v2 = self.v2(x2)
        #attn = torch.einsum('b h m n, b h m c -> b h m n',q,k)
        attn = q * k
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1 = attn * v1#torch.einsum('b h m n, b h m c -> b h m c', attn, v)
        x1 = einops.rearrange(x1, 'b h n c1 -> b n (h c1)')
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        x2 = attn * v2  # torch.einsum('b h m n, b h m c -> b h m c', attn, v)
        x2 = einops.rearrange(x2, 'b h n c1 -> b n (h c1)')
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        return x1,x2, attn


class CrossAttention_Single(nn.Module):
    def __init__(self, dim1,dim2,   num_heads = 8,qkv_bias=False,qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()



        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.k = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim1 % num_heads == 0
        head_dim = dim1 // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim1)
    def forward(self, x1, x2):
        B, N ,C = x1.size()





        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)


        q = q * self.scale
        q = einops.rearrange(q, 'b n  (h1  c1) -> b  h1 n c1', h1 = self.num_heads)
        k = einops.rearrange(k, 'b n  (h1  c2) -> b  h1 n c2', h1 = self.num_heads)
        v = einops.rearrange(v, 'b n  (h1  c1) -> b  h1 n c1', h1 = self.num_heads)



        attn = (q.transpose(-2,-1) @ k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)




        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)




        return x, attn
class CrossAttention(nn.Module):
    def __init__(self, dim,   num_heads = 8,qkv_bias=False,window_size = 8,qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        window_size = to_2tuple(window_size)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x1, x2):
        B, N ,C = x1.size()





        q1, q2 = self.q(x1), self.q(x2)
        k1, k2 = self.k(x1), self.k(x2)
        v1, v2 = self.v(x1), self.v(x2)








        q1, q2 = q1 * self.scale, q2 * self.scale
        q1 = einops.rearrange(q1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
        k1 = einops.rearrange(k1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
        v1 = einops.rearrange(v1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)

        q2 = einops.rearrange(q2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)
        k2 = einops.rearrange(k2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)
        v2 = einops.rearrange(v2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)


        attn1 = (q1 @ k2.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2= (q2 @ k1.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)


        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)


        return x1,x2, attn1, attn2


class CrossAttentionStage2(CrossAttention):
    def __init__(self,dim,   num_heads = 8,qkv_bias=False,window_size = 8,qk_scale=None, attn_drop=0.):
        super().__init__(
            dim = dim,
            num_heads= num_heads,
            qkv_bias = qkv_bias,
            window_size=window_size,
            qk_scale = qk_scale,
            attn_drop = attn_drop
        )

    def forward(self, x1, x2):
        B, N ,C = x1.size()





        q1, q2 = self.q(x1), self.q(x2)
        k1, k2 = self.k(x1), self.k(x2)
        v1, v2 = self.v(x1), self.v(x2)








        q1, q2 = q1 * self.scale, q2 * self.scale
        q1 = einops.rearrange(q1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
        k1 = einops.rearrange(k1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)
        v1 = einops.rearrange(v1, 'b n  (h1  c) -> b  h1 n c', h1 = self.num_heads)

        q2 = einops.rearrange(q2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)
        k2 = einops.rearrange(k2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)
        v2 = einops.rearrange(v2, 'b n  (h1  c) -> b  h1 n c', h1=self.num_heads)


        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2= (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)


        x2 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        x1 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)


        return x1,x2, attn1, attn2




import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodSearch(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, window_size=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 预计算邻域索引（直接存储为张量）
        # self.neighbor_indices = self._precompute_neighbors()
        self.neighbor_norm = nn.LayerNorm(embed_dim)
    def _precompute_neighbors(self, H, W):
        """
        预计算所有像素的邻域索引，并存储为张量格式
        """
        neighbors_list = []
        for idx in range(H * W):
            row, col = idx // W, idx % W
            neighbors = []
            for i in range(-self.window_size, self.window_size + 1):
                for j in range(-self.window_size, self.window_size + 1):
                    r, c = row + i, col + j
                    if 0 <= r < H and 0 <= c < W:
                        neighbors.append(r * W + c)
            neighbors_list.append(neighbors)

        # 转为 PyTorch Tensor，确保索引并行计算
        max_neighbors = max(len(n) for n in neighbors_list)  # 找到最大邻域数
        padded_neighbors = [n + [n[0]] * (max_neighbors - len(n)) for n in neighbors_list]  # 填充
        return torch.tensor(padded_neighbors, dtype=torch.long)  # [N, max_neighbors]

    def forward(self, x1, x2):
        """
        x1, x2: [B, N, D]  ->  B 批量大小, N = H*W, D 特征维度
        """
        B, N, D = x1.shape
        device = x1.device
        H =  W = int(math.sqrt(N))
        neighbor_indices = self._precompute_neighbors(H,W)
        # 获取邻域索引（加速）
        neighbors_idx = neighbor_indices.to(device)  # [N, max_neighbors]

        # 直接索引所有邻域特征 (批量化)
        neighbor_feats = x2[:, neighbors_idx, :]  # [B, N, max_neighbors, D]

        # 获取中心特征
        center_feat = x2.unsqueeze(2)  # [B, N, 1, D]

        # 计算注意力权重 (批量方式)
        attn_weights = F.softmax(torch.einsum('bncd,bnkd->bnkc', center_feat, neighbor_feats) / (D ** 0.5), dim=-1)  # [B, N, max_neighbors, 1]

        # 加权融合
        x2_fused = (neighbor_feats * attn_weights).sum(dim=2)  # [B, N, D]
        x2_fused = self.neighbor_norm(x2_fused)
        # 标准 Cross-Attention (x1 as Query, x2_fused as Key, Value)
        attn_output, attn_map = self.attn(query=x2_fused, key=x1, value=x1)  # [B,N,D]   查询相似 attn_output

        return attn_output








class CrossModalityEncoder(nn.Module):
    def __init__(self, dim, num_heads, window_size = 8,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(attn_drop) if attn_drop > 0. else nn.Identity()
        self.cross_attn1 = nn.ModuleList(
            [
                CrossAttention(dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop),
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop),
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
            ]
        )
        self.cross_attn2 = nn.ModuleList(
            [
                CrossAttentionStage2(dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop),
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop),
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
            ]
        )
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=act_layer, drop=drop)
    def forward(self, x1, x2, return_attention=False):
        y1, y2, _, _ = self.cross_attn1[0](self.norm(x1), self.norm(x2))
        y1,_ = self.cross_attn1[1](y1)
        y2,_ = self.cross_attn1[2](y2)

        y1, y2, _, _ = self.cross_attn2[0](y1, y2)
        y1,_ = self.cross_attn2[1](y1)
        y2,_ = self.cross_attn2[2](y2)


        x1 = x1 + self.drop_path(y1)
        x1 = x1 + self.drop_path(self.mlp1(self.norm1(x1)))

        x2 = x2 + self.drop_path(y2)
        x2 = x2 + self.drop_path(self.mlp2(self.norm2(x2)))

        return x1,x2
class MoEFeatureFusion_v0(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64, drop = 0.):
        super().__init__()
        # gate: 用于为每个“专家”计算权重（softmax over 4）
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4个专家
        )
        # 输出 projection（可选）
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.minus_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    # def forward(self, x1, x2):
    #     x_minus = x1 - x2 #关注变化趋势
    #     x_diff = torch.abs(x1 - x2) #关注变化强度
    #
    #     # 拼接后用于 gating
    #     x_concat = torch.cat([x1, x2, x_minus, x_diff], dim=-1)  # [B, N, 4C]
    #     gate_logits = self.gate(x_concat)  # [B, N, 4]
    #     gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N, 4]
    #
    #     # 组合专家
    #     experts = torch.stack([x1, x2, x_minus, x_diff], dim=-2)  # [B, N, 4, C]
    #     fused = torch.sum(gate_weights.unsqueeze(-1) * experts, dim=-2)  # [B, N, C]
    #
    #     return self.out_proj(fused)
    def forward(self, x1, x2):
        # 基本差异特征
        x_minus = self.minus_proj(x1 - x2)
        x_diff =  self.diff_proj(torch.abs(x_minus))

        # 变化注意力：关注变化显著区域
        sim = F.cosine_similarity(x1, x2, dim=-1)  # [B, N]
        diff_attn = torch.sigmoid(1 - sim).unsqueeze(-1)  #


        # x1_w =  x1 +  x1 * diff_attn
        # x2_w =  x2 +  x2 * diff_attn
        x1_w = x1 * diff_attn
        x2_w = x2 * diff_attn

        # 拼接作为 gating 输入
        x_concat = torch.cat([x1_w, x2_w, x_minus, x_diff], dim=-1)  # [B, N, 4C]

        # Gating 权重
        gate_logits = self.gate(x_concat)  # [B, N, 4]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N, 4]

        # Expert 特征集合
        experts = torch.stack([x1_w, x2_w, x_minus, x_diff], dim=-2)  # [B, N, 4, C]

        # 加权融合
        fused = torch.sum(gate_weights.unsqueeze(-1) * experts, dim=-2)  # [B, N, C]

        return self.out_proj(fused)  # [B, N, C]
#

class MoEFeatureFusion(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64, drop=0.):
        super().__init__()


        self.minus_proj_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop)
        )
        # self.abs_proj_conv = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(drop)
        # )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )



        # 输出映射
        self.out_proj = Mlp(in_features=embed_dim,hidden_features=4 * embed_dim, out_features=embed_dim,drop=drop)


    def forward(self, x1, x2):
        """
        x1, x2: [B, N, C]
        return: [B, N, C]
        """
        B, N, C = x1.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} must be a square patch (H=W=√N), got H*W={H}*{W}"

        # === Step 1: 提取结构感知差异特征 ===
        x_minus = x1 - x2  # [B, N, C]
        x_minus_2d = einops.rearrange(x_minus, 'b (h w) c -> b c h w', h=H)
        x_minus_2d = self.minus_proj_conv(x_minus_2d)  # [B, C, H, W]

        x_minus = einops.rearrange(x_minus_2d, 'b c h w -> b (h w) c')  # [B, N, C]

        # === Step 2: 差异强度特征 ===
        # x_diff = self.abs_proj_conv(torch.abs(x_minus_2d))  # [B, C, H, W]
        # x_diff = einops.rearrange(x_diff, 'b c h w -> b (h w) c')  # [B, N, C]
        x_diff = torch.abs(x_minus)  # [B, C, H, W]
        # # === Step 3: 差异注意力引导 ===
        sim = F.cosine_similarity(x1, x2, dim=-1).unsqueeze(-1)  # [B, N, 1]
        diff_attn = torch.sigmoid(1 - sim)  # [B, N, 1]
        # x1 = x1 - x1 *diff_attn
        # x2 = x2 + x2 *diff_attn
        # x1_w = x1 * (1 - diff_attn) # 弱化 x1
        # x2_w = x2 * (1 + diff_attn)  # 强化 x2

        # === Step 4: 拼接 gating 输入并融合 ===



        x_concat = torch.cat([x1, x2, x_minus, x_diff], dim=-1)  # [B, N, 4C]
        gate_logits = self.gate(x_concat)  # [B, N, 4]
        gate_weights = F.softmax(gate_logits, dim=-1)

        experts = torch.stack([x1, x2, x_minus, x_diff], dim=-2)  # [B, N, 4, C]
        fused = torch.sum(gate_weights.unsqueeze(-1) * experts, dim=-2)  # [B, N, C]
        fused = fused + fused * diff_attn
        return self.out_proj(fused)  # [B, N, C]


class MoEConvFusion4Experts(nn.Module):
    def __init__(self, in_channels, out_dim, num_heads, use_norm=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.in_channels = in_channels  # C
        self.use_norm = use_norm

        # Normalization layers (LayerNorm)
        self.norm = nn.LayerNorm(in_channels) if use_norm else nn.Identity()

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 4, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 4)  # 4 experts
        )

        # Attention and Channel Attention
        self.attn = Attention(in_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop)
        self.ch_attn = ChannelsAttention(in_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop)

        # MLP for output projection
        self.out_proj = Mlp(in_features=in_channels, hidden_features=4 * in_channels, out_features=out_dim,
                            drop=drop)

        # Dropout and Drop path layers for better regularization
        self.drop_path = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 4

        # Split the input into 4 experts: [B, C, H, W]
        x1, x2, x3, x4 = x
        B, C, H, W = x1.shape

        # Rearrange each expert to [B, N, C]
        x1 = einops.rearrange(x1, 'b c h w -> b (h w) c')
        x2 = einops.rearrange(x2, 'b c h w -> b (h w) c')
        x3 = einops.rearrange(x3, 'b c h w -> b (h w) c')
        x4 = einops.rearrange(x4, 'b c h w -> b (h w) c')

        # Apply normalization on each expert
        x1, x2, x3, x4 = map(self.norm, (x1, x2, x3, x4))

        # Gate input: [B, N, 4C]
        gate_input = torch.cat([x1, x2, x3, x4], dim=-1)
        gate_weight = F.softmax(self.gate(gate_input), dim=-1)  # [B, N, 4]

        # MoE fusion: Apply gate weights to fuse the experts
        experts = torch.stack([x1, x2, x3, x4], dim=-2)  # [B, N, 4, C]
        fused = torch.sum(gate_weight.unsqueeze(-1) * experts, dim=-2)  # [B, N, C]

        # Apply Channel Attention after fusion
        fused, ch_attn = self.ch_attn(self.norm(fused))

        # Apply Attention
        fused, sp_attn = self.attn(self.norm(fused))

        # Apply dropout path and MLP
        fused = fused + self.drop_path(fused)
        fused = fused + self.drop_path(self.out_proj(self.norm(fused)))

        # Rearrange back to [B, C, H, W]
        fused = einops.rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)

        return fused










class Cross_Modal_Attention(nn.Module):
    def __init__(self,width,  dim, num_heads, window_size = 8,mlp_ratio=4.,depth= 1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, H = 16, W = 16, feats_fusion = "TBAM", feats_exchange = "CCMAT"):
        super(Cross_Modal_Attention,self).__init__()

        self.feats_exchange = feats_exchange

        self.embed = nn.Sequential(
            nn.Linear(width,dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

        #self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3= nn.LayerNorm(dim)
        self.norm_fuse = nn.LayerNorm(dim)


        self.mlp = Mlp(in_features=dim,hidden_features=dim * 4 ,out_features = dim,act_layer=act_layer,drop = drop)
        #self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=act_layer, drop=drop)


        self.init_attn(dim, num_heads,  window_size, depth,qkv_bias, qk_scale, drop, attn_drop)




        self.feats_fusion = feats_fusion
        if self.feats_fusion == "TBAM":
            self.mlp2 = Mlp(in_features=dim * 4, hidden_features=dim * 4, out_features=dim, act_layer=act_layer,
                            drop=drop)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
            self.ch_attn = ChannelsAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop)
            self.diff_layer = Mlp(dim, hidden_features= dim * 4)
        elif self.feats_fusion == "diff":
            self.fusion = nn.PairwiseDistance(2)
        elif self.feats_fusion == 'concat':
            self.fusion = Mlp(in_features=dim * 2 , hidden_features= dim * 4 ,out_features=dim,act_layer=act_layer,drop = drop)
        elif self.feats_fusion == 'cross':
            self.fusion = CrossAttention_Single(dim = dim,num_heads=24)
        elif self.feats_fusion == 'CBAM':
            self.fusion = CAM_PAM(dim = dim)
        elif self.feats_fusion == 'SCAM':
            self.fusion = NeighborhoodSearch(embed_dim=dim, num_heads=num_heads, window_size=1)
            self.moe = MoEFeatureFusion(embed_dim=dim, hidden_dim=dim, drop=drop)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
            self.ch_attn = ChannelsAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             attn_drop=attn_drop)
            ##self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)

        elif self.feats_fusion == 'MoE':
            self.fusion = MoEFeatureFusion(embed_dim=dim, hidden_dim=dim,drop=drop)
        else:
            raise Exception



    def init_attn(self, dim ,num_heads,window_size = 8,depth = 1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            if self.feats_exchange is None:
                self.blocks = None
                return
            #self.neighbor_fusion = NeighborhoodSearch(embed_dim=dim, num_heads=8, H=16, W=16, window_size=1)
            self.blocks = nn.ModuleList([
                CCMAT(
                    dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop,feats_exchange = self.feats_exchange)
                for _ in range(depth)
            ])
           # self.blocks = None



        # if self.feats_exchange == 'CCMAT':
        #     self.blocks = nn.ModuleList([
        #         Channel_wise_Cross_Modality(
        #             dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #             attn_drop=attn_drop, proj_drop=drop)
        #         for _ in range(depth)
        #     ])
        # elif self.feats_exchange == 'CMAT':
        #     self.blocks =  nn.ModuleList([
        #         CrossModalityEncoder(
        #         dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         attn_drop=attn_drop)
        #         for _ in range(depth)
        #     ])
        # else:
        #     raise "{} not supported.".format(self.feats_exchange)
    def forward(self, x1,x2, return_attention =False):

        x1 = self.embed(x1)
        x2 = self.embed(x2)

        if self.blocks is not None:
            for i, blk in enumerate(self.blocks):
                #x1, x2 = self.neighbor_fusion(x1, x2)
                x1, x2 = blk(x1, x2)[:2]

        #compute x_diff and x_minus  according x1 and x2




        if self.feats_fusion == 'TBAM':

            x_minus =  x1 - x2
            x_diff = torch.abs(x1 - x2)
            #用 x_diff 去加权 x1 和 x2 , 保留x1 和 x2 差异化特征

            #x = self.mlp2(torch.cat((x1, x2, x_minus, x_diff), dim=-1))  #建筑物和背景色相似，但是有结构？
            x = self.mlp2(torch.cat((x1, x2, x_minus,x_diff), dim=-1))
            #x = self.mlp2(torch.cat((x1, x2), dim=-1))
            x, ch_attn = self.ch_attn(self.norm(x))
            y, attn = self.attn(self.norm_fuse(x))
            if return_attention:
                return attn

            x = x + self.drop_path(y)

            x = x + self.drop_path(self.mlp(self.norm1(x)))
        elif self.feats_fusion == 'diff':
            x = torch.abs(x1 - x2)
        elif self.feats_fusion == 'concat':
            x  = self.fusion(torch.cat((x1,x2),dim=1))

        elif self.feats_fusion == "CBAM":
            x = self.mlp2(torch.cat((x1, x2), dim=-1))
            x = self.fusion(x)
            x = einops.rearrange(x,'b c h w -> b (h w) c')
        elif self.feats_fusion == 'SCAM':
            x_21 = self.fusion(x2, x1)
            x2 = x2 + self.drop_path(x_21)
            x2 = x2 + self.drop_path(self.mlp(self.norm_fuse(x2)))
            x = self.moe(self.norm1(x1), self.norm1(x2))
            #x = self.moe(x1, x2)
            x, ch_attn = self.ch_attn(self.norm1(x))
            y, attn = self.attn(self.norm2(x))
            x = x + self.drop_path(y)
            #x = x + self.drop_path(self.mlp1(self.norm3(x)))

        return  x #torch.cat((x,x_diff), dim = -1)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):

        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out
if __name__ == '__main__':

    from lib.utils.tools.configer import Configer
    x1 = torch.randn((1, 1024, 768),dtype=torch.float)
    x2 = torch.randn((1, 1024, 768), dtype=torch.float)
    model = CrossAttention(dim=768,num_heads=6)

    out = model(x1, x2)
    # model = Cross_ChannelsAttention(dim=768,num_heads=6)
    # out =model(x1,x2)

    models1 = Attention(dim=768,num_heads=6)
    out = models1(x1)
    print(out.shape)


