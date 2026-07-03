import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class BasicSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, kernel_disp=17, stride=1):
        super(BasicSeparableConv3D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size//2, kernel_size//2),  stride=(1, stride, stride), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0),  stride=(stride, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):# (32x, 16x), (16x, 8x), (8x,4x)...
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv
    
class ResnetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out
  
class ResnetBasicBlock3D(nn.Module):
  """
  https://arxiv.org/abs/2501.09898
  """
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out
  
class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise-separable Convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))

class LightResBlock3D(nn.Module):
    """ResnetBasicBlock3D with depthwise-separable convs."""
    def __init__(self, planes, kernel_size=3, padding=1,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3d(planes, planes, kernel_size, padding=padding),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv3d(planes, planes, kernel_size, padding=padding),
            norm_layer(planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
    
class PositionalEmbedding(nn.Module):
  """
  https://github.com/NVlabs/FoundationStereo/blob/master/core/submodule.py
  """
  def __init__(self, d_model, max_len=12):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe, persistent=False)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]

class FlashMultiheadAttention(nn.Module):
    """
    https://github.com/NVlabs/FoundationStereo/blob/master/core/submodule.py
    """
    def __init__(self, embed_dim, att_head_latent_dim, n_head):
        super().__init__()
        self.num_heads = n_head
        self.embed_dim = embed_dim
        self.head_dim = att_head_latent_dim // n_head
        assert self.head_dim * n_head == att_head_latent_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, att_head_latent_dim)
        self.k_proj = nn.Linear(embed_dim, att_head_latent_dim)
        self.v_proj = nn.Linear(embed_dim, att_head_latent_dim)
        self.out_proj = nn.Linear(att_head_latent_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = F.scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.reshape(B,L,-1)
        #print(f"Attention Pre-Output Projection: {attn_output.shape}")
        output = self.out_proj(attn_output)
        #print(f"Attention Post-Output Projection: {output.shape}")
        return output

class FlashAttentionTransformerEncoderLayer(nn.Module):
    """
    https://github.com/NVlabs/FoundationStereo/blob/master/core/submodule.py
    """
    def __init__(self, embed_dim, att_head_latent_dim, n_head, mlp_latent_dim, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, att_head_latent_dim, n_head)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, mlp_latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_latent_dim, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class DisparityTransformer(nn.Module):
    """
    Adapted from https://arxiv.org/abs/2501.09898.
    https://github.com/NVlabs/FoundationStereo/blob/master/core/submodule.py
    """
    def __init__(self, in_channels, n_encoder_layers, embed_dim, att_head_latent_dim, n_head, mlp_latent_dim, pe_max_len=192/(4*4)):
        super().__init__()
        
        self.conv_patch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm3d(in_channels),
        )
        self.layers = nn.ModuleList([])
        for _ in range(n_encoder_layers):
            self.layers.append(FlashAttentionTransformerEncoderLayer(
                embed_dim=embed_dim,
                att_head_latent_dim=att_head_latent_dim,
                n_head=n_head,
                mlp_latent_dim=mlp_latent_dim,
                dropout=0.1
                )
            )
        self.pos_embed0 = PositionalEmbedding(in_channels, max_len=pe_max_len)

    def forward(self, cv, window_size=(-1,-1)):
        """
        @cv: `[B, C, D, H, W]`
        """
        x = self.conv_patch(cv)
        B, C, D, H, W = x.shape
        x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
        x = self.pos_embed0(x, resize_embed=False)
        for i in range(len(self.layers)):
            x = self.layers[i](x, window_size)
        x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)
        x = F.interpolate(x, scale_factor=4.0, mode='trilinear', align_corners=False)
        return x

class FastResidualBlock_32_16(nn.Module):
    """
    Based on https://arxiv.org/abs/2512.11130.
    """
    def __init__(self, in_channels, out_channels, feat_channels):
        super(FastResidualBlock_32_16, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                )
        )

        self.feat_att = FeatureAtt(cv_chan=out_channels, feat_chan=feat_channels)

        self.conv1 = BasicConv(
                in_channels=out_channels,
                out_channels=out_channels,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        
        self.conv2 = BasicSeparableConv3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )
        
        self.conv3 = BasicConv(
                in_channels=out_channels,
                out_channels=out_channels,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, x_up, feats):
        x = x + self.upsample(x_up)
        x = self.feat_att(x, feats)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    
class FasterResidualBlock_32_16(nn.Module):
    """
    Based on https://arxiv.org/abs/2512.11130.
    """
    def __init__(self, in_channels, out_channels, feat_channels):
        super(FasterResidualBlock_32_16, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                )
        )

        self.feat_att = FeatureAtt(cv_chan=out_channels, feat_chan=feat_channels)

        self.conv1 = BasicSeparableConv3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )

    def forward(self, x, x_up, feats):
        x = x + self.upsample(x_up)
        x = self.feat_att(x, feats)
        x = self.conv1(x)

        return x
        

class FastResidualBlock_16_08(nn.Module):
    """
    Based on https://arxiv.org/abs/2512.11130.
    """
    def __init__(self, in_channels, out_channels, feat_channels):
        super(FastResidualBlock_16_08, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=4,
                stride=2,
                padding=1,
                )
        )

        self.conv1 = BasicConv(
                in_channels=out_channels,
                out_channels=out_channels,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        
        self.feat_att = FeatureAtt(cv_chan=out_channels, feat_chan=feat_channels)

        self.conv2 = BasicSeparableConv3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )

    def forward(self, x, x_up, feats):
        x = x + self.upsample(x_up)
        x = self.conv1(x)
        x = self.feat_att(x, feats)
        x = self.conv2(x)

        return x
    
class FasterResidualBlock_16_08(nn.Module):
    """
    Based on https://arxiv.org/abs/2512.11130.
    """
    def __init__(self, in_channels, out_channels, feat_channels):
        super(FasterResidualBlock_16_08, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=4,
                stride=2,
                padding=1,
                )
        )

        self.feat_att = FeatureAtt(cv_chan=out_channels, feat_chan=feat_channels)

        self.conv1 = BasicSeparableConv3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )

    def forward(self, x, x_up, feats):
        x = x + self.upsample(x_up)
        x = self.feat_att(x, feats)
        x = self.conv1(x)

        return x

class FastResidualBlock_04(nn.Module):
    """
    Based on https://arxiv.org/abs/2512.11130.
    """
    def __init__(self, in_channels, out_channels):
        super(FastResidualBlock_04, self).__init__()
        self.upsample = nn.Sequential(
            DisparityTransformer(
                in_channels=in_channels,
                n_encoder_layers=1,
                embed_dim=in_channels,
                att_head_latent_dim=in_channels,
                n_head=4,
                mlp_latent_dim=in_channels*4,
                pe_max_len=192// (4*4))
        )
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x, x_up):
        x = x + self.upsample(x_up)
        x = self.relu(self.norm(self.conv(x)))
        return x

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

#@torch.compile(mode="reduce-overhead")
def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(prob, maxdisp, interval):
    assert len(prob.shape) == 4
    disp_values = torch.arange(0, maxdisp, interval, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp//interval, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)

def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape       
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    disp = (disp_unfold*up_weights).sum(dim=1,keepdim=True)      
    return disp

def norm_correlation(fea1, fea2):
    """
    https://arxiv.org/pdf/2209.12699
    """
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    """
    https://arxiv.org/pdf/2209.12699
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

class Aggregation2D(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation2D, self).__init__()
        
        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [ConvNeXtBlock(in_channels) for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2),
        )

        conv2_add = [ConvNeXtBlock(in_channels * 2) for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = nn.Sequential(
            LayerNorm(in_channels * 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=2, stride=2),
        )

        conv4_add = [ConvNeXtBlock(in_channels * 4) for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False)
        )

        self.redir1 = ConvNeXtBlock(in_channels)
        self.redir2 = ConvNeXtBlock(in_channels * 2)

        if self.left_att:
            self.att4 = AttentionModule2D(in_channels, backbone_channels[0])
            self.att8 = AttentionModule2D(in_channels * 2, backbone_channels[1])
            self.att16 = AttentionModule2D(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        x_4 = self.conv0(x)
        if self.left_att:
            x_4 = self.att4(x_4, features_left[0])

        x_8 = self.conv1(x_4)
        x_8 = self.conv2(x_8)
        if self.left_att:
            x_8 = self.att8(x_8, features_left[1])

        x_16 = self.conv3(x_8)
        x_16 = self.conv4(x_16)
        if self.left_att:
            x_16 = self.att16(x_16, features_left[2])

        x_8 = F.relu(self.upconv1(x_16) + self.redir2(x_8), inplace=True)
        x_4 = F.relu(self.upconv2(x_8) + self.redir1(x), inplace=True)

        return x_4, x_8, x_16
        #return self.out_act(self.out_proj4(x_4)), self.out_act(self.out_proj8(x_8)), self.out_act(self.out_proj16(x_16))

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return x + input

class AttentionModule2D(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost
    
class GroupedAttentionModule2D(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        # To Prevent Feature Channels and Disparity from mixing
        #self.conv0 = nn.Conv2d(img_feat_dim, img_feat_dim, 1, groups=dim)
        dim_out = dim
        dim = img_feat_dim

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim_out, 1, groups=dim_out)
        self.conv4 = nn.Conv2d(dim_out, dim_out, 1)

    def forward(self, cost, x):
        #attn = self.conv0(x)

        attn_0 = self.conv0_1(x)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(x)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(x)
        attn_2 = self.conv2_2(attn_2)

        attn = x + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn) # Project to Disparity-Space
        attn = self.conv4(attn).unsqueeze(1) # Cross Disparity

        return attn * cost


@torch.compile
def build_gwc_volume_optimized_pytorch1(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int, normalize=True):
  """
  https://github.com/NVlabs/Fast-FoundationStereo/blob/master/core/submodule.py
  """
  dtype = refimg_fea.dtype
  B, C, H, W = refimg_fea.shape
  channels_per_group = C // num_groups

  ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
  padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))
  unfolded_target = padded_target.unfold(3, W, 1)
  target_volume = torch.flip(unfolded_target, [3]).permute(0, 1, 3, 2, 4)
  ref_volume = ref_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
  target_volume = target_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
  if normalize:
    ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
    target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)

  cost_volume = (ref_volume * target_volume).sum(dim=2)

  return cost_volume.contiguous()

if triton is not None and torch.cuda.is_available():
  @triton.autotune(configs=[
    triton.Config({'BLOCK_C':4,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':8,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':16,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':64,'BLOCK_W':128,'BLOCK_D':8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_C':128,'BLOCK_W':64,'BLOCK_D':8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_C':128,'BLOCK_W':128,'BLOCK_D':8}, num_warps=8, num_stages=2),
  ], key=['C','W','D','G','K','NORMALIZE'])
  @triton.jit
  def _gwc_triton_kernel(ref_ptr, tar_ptr, ref_norm_ptr, tar_norm_ptr, out_ptr, BH, C, W, D: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                         stride_rn, stride_rw, stride_rc, stride_tn, stride_tw, stride_tc,
                         stride_nn, stride_ng, stride_nw,
                         stride_on, stride_og, stride_od, stride_ow,
                         NORMALIZE: tl.constexpr,
                         BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_D: tl.constexpr):
    pid0 = tl.program_id(0)
    db = tl.program_id(1)
    wb = tl.program_id(2)
    bh = pid0 // G
    g = pid0 % G
    w_off = wb*BLOCK_W + tl.arange(0, BLOCK_W)
    d_off = db*BLOCK_D + tl.arange(0, BLOCK_D)
    w_mask = w_off < W
    w_src = w_off[None, :] - d_off[:, None]
    td_mask = (w_src >= 0) & w_mask[None, :]
    acc = tl.zeros((BLOCK_D, BLOCK_W), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_C):
      k_off = k0 + tl.arange(0, BLOCK_C)
      k_mask = k_off < K
      c_idx = g*K + k_off
      ref_ptrs = ref_ptr + bh*stride_rn + w_off[None, :]*stride_rw + c_idx[:, None]*stride_rc
      ref_vals = tl.load(ref_ptrs, mask=k_mask[:, None] & w_mask[None, :], other=0.).to(tl.float32)
      tar_ptrs = tar_ptr + bh*stride_tn + w_src[None, :, :]*stride_tw + c_idx[:, None, None]*stride_tc
      tar_vals = tl.load(tar_ptrs, mask=k_mask[:, None, None] & td_mask[None, :, :], other=0.).to(tl.float32)
      acc += tl.sum(tar_vals * ref_vals[:, None, :], axis=0)

    if NORMALIZE:
      norm_offset = bh*stride_nn + g*stride_ng
      ref_norm = tl.load(ref_norm_ptr + norm_offset + w_off*stride_nw, mask=w_mask, other=1.0).to(tl.float32)
      tar_norm = tl.load(tar_norm_ptr + norm_offset + w_src*stride_nw, mask=td_mask, other=1.0).to(tl.float32)
      denom = (ref_norm[None, :] * tar_norm) + 1e-5
      acc = acc / denom
    out_ptrs = out_ptr + bh*stride_on + g*stride_og + d_off[:, None]*stride_od + w_off[None, :]*stride_ow
    tl.store(out_ptrs, acc, mask=w_mask[None, :])

@torch.no_grad()
def build_gwc_volume_triton(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int, normalize=True):
  """
  https://github.com/NVlabs/Fast-FoundationStereo/blob/master/core/submodule.py
  """
  if triton is None:
    raise RuntimeError('Triton is not available. Please install triton to use build_gwc_volume_triton.')
  B, C, H, W = refimg_fea.shape
  assert maxdisp > 0 and C % num_groups == 0
  K = C // num_groups
  in_dtype = refimg_fea.dtype if refimg_fea.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

  if normalize:
    ref_norm = refimg_fea.float().view(B, num_groups, K, H, W).norm(dim=2)
    tar_norm = targetimg_fea.float().view(B, num_groups, K, H, W).norm(dim=2)
    ref_norm = ref_norm.permute(0, 2, 1, 3).reshape(B*H, num_groups, W).to(in_dtype).contiguous()
    tar_norm = tar_norm.permute(0, 2, 1, 3).reshape(B*H, num_groups, W).to(in_dtype).contiguous()
  else:
    # Dummy tensors; kernel won't read them when NORMALIZE=False
    ref_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)
    tar_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)

  ref = refimg_fea.to(in_dtype)
  tar = targetimg_fea.to(in_dtype)
  ref_bhwc = ref.permute(0, 2, 3, 1).view(B * H, W, C).contiguous()
  tar_bhwc = tar.permute(0, 2, 3, 1).view(B * H, W, C).contiguous()
  out_bhw = torch.empty((B * H, num_groups, maxdisp, W), device=ref.device, dtype=in_dtype)
  BH = B * H
  D_eff = min(maxdisp, W)
  grid = lambda META: (BH * num_groups, triton.cdiv(D_eff, META['BLOCK_D']), triton.cdiv(W, META['BLOCK_W']))
  _gwc_triton_kernel[grid](ref_bhwc, tar_bhwc, ref_norm, tar_norm, out_bhw, BH, C, W, D_eff, num_groups, K,
                           ref_bhwc.stride(0), ref_bhwc.stride(1), ref_bhwc.stride(2),
                           tar_bhwc.stride(0), tar_bhwc.stride(1), tar_bhwc.stride(2),
                           ref_norm.stride(0), ref_norm.stride(1), ref_norm.stride(2),
                           out_bhw.stride(0), out_bhw.stride(1), out_bhw.stride(2), out_bhw.stride(3),
                           NORMALIZE=normalize)
  if D_eff < maxdisp: out_bhw[:, :, D_eff:, :] = 0
  volume = out_bhw.view(B, H, num_groups, maxdisp, W).permute(0, 2, 3, 1, 4).contiguous()
  return volume

@torch.compile
def build_concat_volume_optimized_pytorch1(refimg_fea, targetimg_fea, maxdisp:int):
    """
    https://github.com/NVlabs/Fast-FoundationStereo/blob/master/core/submodule.py
    """
    B, C, H, W = refimg_fea.shape

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))  # (B, C, H, W + maxdisp - 1)
    unfolded_target = padded_target.unfold(dimension=3, size=W, step=1)  # (B, C, H, maxdisp, W)
    target_volume = torch.flip(unfolded_target, [3]).permute(0, 1, 3, 2, 4)
    volume = torch.cat((ref_volume, target_volume), dim=1)
    return volume.contiguous()