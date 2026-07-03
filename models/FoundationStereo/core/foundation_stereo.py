# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from models.FoundationStereo.core.update import *
from models.FoundationStereo.core.extractor import *
from models.FoundationStereo.core.geometry import Combined_Geo_Encoding_Volume
from models.FoundationStereo.core.submodule import *
from models.FoundationStereo.core.utils.utils import *
from models.FoundationStereo.core.utils import *
import time,huggingface_hub
from argparse import Namespace
from pathlib import Path
from omegaconf import OmegaConf
from utils.distillation_tools import Paraphraser
from torch.amp import autocast

def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)

        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        self.cnet = ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature(args)
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )


        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )


        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        if args.features and args.factor_transfer:
            self.feature_paraphraser = Paraphraser(in_channels=self.feature.d_out[0], latent_dim=96, use_bn=False)

        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx


    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()


    def forward(self, image1, image2, iters=32, test_mode=False, teach_mode=False, low_memory=False, s_init_disp=None, train_feat_paraphraser=False):
        """ Estimate disparity between pair of frames """
        
        if teach_mode:
            teacher_outputs = {}

        B = len(image1)
        low_memory = low_memory or (self.args.get('low_memory', False))
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = self.stem_2(image1)

            if train_feat_paraphraser:
                return [features_left[0], features_right[0]], [self.feature_paraphraser(features_left[0]), self.feature_paraphraser(features_right[0])]
            
            if teach_mode and self.args.features:
                resolutions = ['4x', '8x', '16x', '32x']
                for res, feat_left, feat_right in zip(resolutions, features_left, features_right):
                    teacher_outputs[f"feat_{res}_left"] = self.feature_paraphraser(feat_left, return_latent=True).detach() if res=='4x' and self.args.factor_transfer else feat_left.detach()
                    teacher_outputs[f"feat_{res}_right"] = self.feature_paraphraser(feat_right, return_latent=True).detach() if res=='4x' and self.args.factor_transfer else feat_right.detach()

            gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
            del left_tmp, right_tmp
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume) # [B, 28, 104, H//4, W//4]

            if teach_mode and self.args.cost_volume:
                teacher_outputs["cost_volume"] = comb_volume[:,:,0:48,:,:].detach() # Truncation: [B, 28, 48, H//4, W//4]
            
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left) 

            if teach_mode and self.args.agg_cost_volume:
                teacher_outputs["agg_cost_volume"] = comb_volume[:,:,0:48,:,:].detach() # Truncation: [B, 28, 48, H//4, W//4]
            
            init_disp_logits = self.classifier(comb_volume).squeeze(1) # Logits: [B, 104, H//4, W//4]

            if teach_mode and self.args.init_disp_logits:
                teacher_outputs["init_disp_logits"] = init_disp_logits.detach()
            
            prob = F.softmax(init_disp_logits, dim=1)  #(B, max_disp, H, W)
            
            init_disp = disparity_regression(prob, self.args.max_disp//4)  # Weighted sum of disparity

            cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list(cnet_list)

            net_list = [torch.tanh(x[0]) for x in cnet_list]   # Hidden information
            inp_list = [torch.relu(x[1]) for x in cnet_list]   # Context information list of pyramid levels
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

            if teach_mode and self.args.init_hidden_state:
                teacher_outputs["init_hidden_state"] = net_list[0].detach()


        geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = s_init_disp.float() if self.args.use_s_init_disp else init_disp.float()
        disp_preds = []

        # GRUs iterations to update disparity (1/4 resolution)
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)
            
            with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net_list, mask_feat_4, delta_disp, update_block_outputs = self.update_block.forward(net_list, inp_list, geo_feat, disp, att, itr)
            
            if teach_mode:
                teacher_outputs = {**teacher_outputs, **update_block_outputs}
                
            disp = disp + delta_disp.float()

            if teach_mode and self.args.refined_disp and itr in self.args.update_block_iters:
                teacher_outputs[f"iter_{itr}_refined_disp"] = disp.squeeze(1).detach()

            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)

        if test_mode and not teach_mode:
            return disp_up
        
        if test_mode and teach_mode:
            return teacher_outputs
            
        return init_disp, disp_preds


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
      B,_,H,W = image1.shape
      img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
      img1_small, img2_small = padder.pad(img1_small, img2_small)
      disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
      disp_small = padder.unpad(disp_small.float())
      disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
      disp_small_up = disp_small_up.clip(0, None)

      padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
      image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
      disp_small_up += padder._pad[0]
      init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
      disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
      disp = padder.unpad(disp.float())
      return disp

def init_foundation_stereo(args=None, use_gpu=True):
    """Initialises Foundation Stereo."""
    checkpoint_path = Path(__file__).parent.parent / "pretrained" / "23-51-11" / "model_best_bp2.pth"    

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint Path not found: {checkpoint_path}")

    foundation_stereo_config = Namespace(
        ckpt_dir=checkpoint_path,
        corr_implementation="reg",
        corr_levels=2,
        corr_radius=4,
        hidden_dims=[128, 128, 128],
        max_disp=416,
        mixed_precision=True,
        precision_dtype=args.precision_dtype if args and hasattr(args, "precision_dtype") else 'float16',
        n_downsample=2,
        n_gru_layers=3,
        valid_iters=args.foundation_stereo_valid_iters if args and hasattr(args, "foundation_stereo_valid_iters") else 32,
        vit_size="vitl",
        # Knowledge Distillation
        being_taught=args.being_taught if args and hasattr(args, "being_taught") else False,
        features=args.features if args and hasattr(args, "features") else False,
        factor_transfer=args.factor_transfer if args and hasattr(args, "factor_transfer") else False,
        cost_volume=args.cost_volume if args and hasattr(args, "cost_volume") else False,
        agg_cost_volume=args.agg_cost_volume if args and hasattr(args, "agg_cost_volume") else False,
        init_disp_logits=args.init_disp_logits if args and hasattr(args, "init_disp_logits") else False,
        init_disp=args.init_disp if args and hasattr(args, "init_disp") else False,
        use_s_init_disp=args.use_s_init_disp if args and hasattr(args, "use_s_init_disp") else False,
        init_hidden_state=args.init_hidden_state if args and hasattr(args, "init_hidden_state") else False,
        update_block_iters=args.update_block_iters if args and hasattr(args, "update_block_iters") else [],
        hidden_state=args.hidden_state if args and hasattr(args, "hidden_state") else False,
        mask=args.mask if args and hasattr(args, "mask") else False,
        delta_disp=args.delta_disp if args and hasattr(args, "delta_disp") else False,
        refined_disp=args.refined_disp if args and hasattr(args, "refined_disp") else False,
    )
    
    cfg = OmegaConf.create(vars(foundation_stereo_config))

    model = FoundationStereo(cfg) # init on cpu

    if args is not None and hasattr(args, "rank") and args.rank == 0:
        print(f"Loading Foundation Stereo checkpoint '23-51-11/model_best_bp2.pth'...")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False) # load to cpu
    model.load_state_dict(ckpt['model'], strict=False)

    if args is not None and hasattr(args, "rank") and args.rank == 0:
        print("Done loading Foundation Stereo checkpoint '23-51-11/model_best_bp2.pth'...")
    
    model.eval()

    # Freeze all parameters
    for param in model.parameters(): 
        param.requires_grad = False

    if use_gpu:
        if args is not None and hasattr(args, "device"):
            DEVICE = torch.device(args.device)
        else:
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    model = model.to(DEVICE) # move to cuda if available

    return model