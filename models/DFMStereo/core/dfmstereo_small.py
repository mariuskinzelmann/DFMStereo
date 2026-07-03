import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.DFMStereo.core.update import BasicUpdateBlock
from models.DFMStereo.core.extractor import Feature_EdgeNext
from models.DFMStereo.core.geometry import Combined_Geo_Encoding_Volume
from models.DFMStereo.core.submodule import *

from pathlib import Path
from argparse import Namespace

try:
    from torch.amp import autocast
except ImportError:
    class autocast:
        def __init__(self, device_type=None, enabled=True, dtype=None):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()
    
class lite_cost_agg_3d(nn.Module):
    """
    Based on https://arxiv.org/abs/2511.16555.
    """
    def __init__(self, channels, max_disp, n_heads, left_att, backbone_channels):
        super().__init__()
        self.n_heads  = n_heads
        self.max_disp = max_disp

        self.cost_agg_3d = nn.Sequential(
            BasicConv(channels, channels*4, is_3d=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels*4, channels*4, is_3d=True, kernel_size=3, stride=1, padding=1),
            BasicConv(channels*4, channels, is_3d=True, kernel_size=3, stride=1, padding=1)
        )
        
        self.agg2d = Aggregation2D(
            in_channels=max_disp,
            left_att=left_att,
            blocks=[4, 8, 16],
            expanse_ratio=4,
            backbone_channels=backbone_channels
        )

    def forward(self, cv, features_left):
        x = self.cost_agg_3d(cv).squeeze(1)
        x4, x8, x16 = self.agg2d(x, features_left)
        
        return x4, x8, x16
        
class att_hourglass(nn.Module):
    def __init__(self, in_channels, max_disp=192):
        super(att_hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),)
                                    
        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),)                             

        self.conv2_up = BasicConv(
                in_channels=in_channels*4,
                out_channels=in_channels*2,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=4,
                stride=2,
                padding=1,
                )
        
        self.conv1_up = BasicConv(
                in_channels=in_channels*2,
                out_channels=in_channels,
                deconv=True,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=4,
                stride=2,
                padding=1,
                )
        
        self.conv2_agg = BasicSeparableConv3D(
                in_channels=in_channels*4,
                out_channels=in_channels*4,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )
        
        self.conv1_agg = BasicSeparableConv3D(
                in_channels=in_channels*2,
                out_channels=in_channels*2,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )
        
        self.conv0_agg = BasicSeparableConv3D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                kernel_disp=9,
                stride=1,
            )
                
        self.ref_att_04 = GroupedAttentionModule2D(max_disp//4, max_disp//4)
        self.ref_att_08 = GroupedAttentionModule2D(max_disp//8, max_disp//2)
        self.ref_att_16 = GroupedAttentionModule2D(max_disp//16, max_disp)

    def forward(self, x, att_4, att_8, att_16):
        conv0 = x

        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.ref_att_16(conv2, att_16)
        conv2 = self.conv2_agg(conv2)

        conv2 = self.conv2_up(conv2)
        conv1 = conv1 + conv2
        conv1 = self.ref_att_08(conv1, att_8)
        conv1 = self.conv1_agg(conv1)

        conv1 = self.conv1_up(conv1)
        conv0 = conv0 + conv1
        conv0 = self.ref_att_04(conv0, att_4)
        conv0 = self.conv0_agg(conv0)

        return conv0

class DFMStereo_Small(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.amp_dtype = getattr(torch, args.precision_dtype, torch.float16)
        self.dtype = torch.float32
        
        # --- Feature Backbone ----
        if args.feature_backbone == 'edgenext':
            self.feature = Feature_EdgeNext()
        elif args.feature_backbone == 'mobilenetv2':
            raise NotImplementedError()
        
        # --- Update Block ---
        hidden_dim = args.hidden_dim
        context_dim = args.context_dim
        self.context_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),   # 1/2
            nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 1/4
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            )
        self.context_fuse = nn.Sequential(
            nn.Conv2d(64 + 96, 128, kernel_size=1),
            ResnetBasicBlock(128, 128, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d),
            ResnetBasicBlock(128, 128, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d),
            )
        self.hnet = nn.Sequential(
            BasicConv_IN(128, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False)
            )
        self.cnet = BasicConv_IN(128, context_dim, kernel_size=3, stride=1, padding=1)
        self.context_zqr_conv = nn.Conv2d(context_dim, context_dim*3, 3, padding=3//2)

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=args.hidden_dim)

        # --- Up Sampling ---
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)


        # --- Cost Volume ---
        proj_dim = self.args.concat_groups
        self.proj_concat = nn.Conv2d(96, proj_dim, kernel_size=1, padding=0)
        volume_dim = self.args.volume_dim

        # Generate 3 Attention Maps from a Correlation Volume at Resolutions: 16x, 8x, 4x.
        # Attention Maps are generated from LiteAnyStereo-Style Cost Aggregation Module.
        # Attention Maps are used as Guidance for Concat Volume Upsampling.
        self.concat_cost_agg = att_hourglass(2 * proj_dim) 
        self.corr_cost_agg = lite_cost_agg_3d(1, self.args.max_disp // 4, n_heads=1, left_att=True, backbone_channels=self.feature.d_out)

        volume_dim = 2 * proj_dim
        
        self.classifier = nn.Sequential(
            nn.Conv3d(volume_dim, volume_dim//2, kernel_size=1, stride=1, padding=0, bias=True),
            ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3)
            )
        
        self.feature_extraction_modules = [
            self.feature,
        ]

        self.cost_filtering_modules = [
            self.proj_concat,
            self.classifier,
            self.stem_2,
            self.spx,
            self.spx_2,
            self.spx_4,
        ]

        self.disparity_refinement_modules = [
            self.context_stem,
            self.context_fuse,
            self.hnet,
            self.cnet,
            self.context_zqr_conv,
            self.update_block,
            self.spx_2_gru,
            self.spx_gru,
        ]
        
        if args.being_taught and args.features:
            if args.factor_transfer and not args.teacher == 'fast_foundation_stereo':
                self.feat_trans_4x = BasicConv(in_channels=96, out_channels=96, bn=False, kernel_size=1, padding=0)
            else:
                self.feat_trans_4x = nn.Conv2d(96, 224, kernel_size=1, stride=1, padding=0, bias=True)
            self.feature_extraction_modules.append(self.feat_trans_4x)

        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.register_buffer("dx", dx)

    def freeze_modules(self, module_list):
        for module in module_list:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

    def unfreeze_modules(self, module_list):
        for module in module_list:
            for p in module.parameters():
                p.requires_grad = True
            module.train()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred)
        return up_disp


    def forward(self, image1, image2, iters=8, test_mode=False, being_taught=False, optimise_volume_build=False):
        """ Estimate disparity between pair of frames """
        if not test_mode:
            student_outputs = {}

        B = len(image1)
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            features = self.feature(torch.cat([image1, image2], dim=0))

            features_left = [f[:B].contiguous() for f in features]
            features_right = [f[B:].contiguous() for f in features]
            stem_2x = self.stem_2(image1) 

            if being_taught and self.args.features:
                resolutions = ['4x', '8x', '16x', '32x']
                for res, feat_left, feat_right in zip(resolutions, features_left, features_right):
                    if res == '4x':
                        student_outputs[f"feat_{res}_left"] = self.feat_trans_4x(feat_left)
                        student_outputs[f"feat_{res}_right"] = self.feat_trans_4x(feat_right)
                    else:
                        student_outputs[f"feat_{res}_left"] = feat_left
                        student_outputs[f"feat_{res}_right"] = feat_right

            if optimise_volume_build:
                corr_volume = build_gwc_volume_optimized_pytorch1(features_left[0], features_right[0], self.args.max_disp//4, num_groups = 1, normalize=False) # [B, 1, D//4, H//4, W//4]
            else:
                corr_volume = build_correlation_volume(features_left[0], features_right[0], self.args.max_disp//4) # [B, 1, D//4, H//4, W//4]

            att_04, att_08, att_16 = self.corr_cost_agg(corr_volume, features_left) # Generate Attention Maps at
            
            proj_left = self.proj_concat(features_left[0])
            proj_right = self.proj_concat(features_right[0])
            if optimise_volume_build:
                concat_volume = build_concat_volume_optimized_pytorch1(proj_left, proj_right, self.args.max_disp//4) # [B, 28, D//4, H//4, W//4]
            else:
                concat_volume = build_concat_volume(proj_left, proj_right, self.args.max_disp//4)
            comb_vol = self.concat_cost_agg(concat_volume, att_04, att_08, att_16)

            if being_taught and self.args.agg_cost_volume:
                student_outputs["agg_cost_volume"] = comb_vol

            init_disp_logits = self.classifier(comb_vol).squeeze(1) # [B, MaxDisp//4, H//4, W//4]

            if being_taught and self.args.init_disp_logits:
                student_outputs["init_disp_logits"] = init_disp_logits

            # Init disp from geometry encoding volume
            prob = F.softmax(init_disp_logits, dim=1) # [B, MaxDisp//4, H//4, W//4]
            init_disp = disparity_regression(prob, self.args.max_disp//4, 1) # [B, H//4, W//4]
            
            if being_taught and self.args.init_disp:
                student_outputs["init_disp"] = init_disp.detach()

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            stem_ctx = self.context_stem(image1)    # [B, 64, H/4, W/4]
            ctx_feat = torch.cat([stem_ctx, features_left[0]], dim=1)   # [B, 160, H/4, W/4]
            ctx_feat = self.context_fuse(ctx_feat)  # [B, 128, H/4, W/4]

            net = torch.tanh(self.hnet(ctx_feat)) # [B, hidden_dim, H/4, W/4]
            context = self.cnet(ctx_feat)   # [B, context_dim, H/4, W/4]
            context = list(self.context_zqr_conv(context).split(split_size=self.args.context_dim, dim=1))

            if being_taught and self.args.init_hidden_state: # Only use with Residual Update Block
                student_outputs["init_hidden_state"] = net.detach()

        geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_vol.float(), num_levels=self.args.corr_levels, dx=self.dx)
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        disp_preds = []

        del comb_vol, features_left, features_right, att_04, att_08, att_16, corr_volume

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)

            with autocast(device_type='cuda', enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                net, mask_feat_4, delta_disp, update_block_outputs = self.update_block(net, context, geo_feat, disp, itr, being_taught)
            
            if being_taught:
                student_outputs = {**student_outputs, **update_block_outputs}
            
            del update_block_outputs

            disp = disp + delta_disp

            if being_taught and self.args.refined_disp and itr in self.args.update_block_iters:
                student_outputs[f"iter_{itr}_refined_disp"] = disp.squeeze(1)

            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp_up = context_upsample(init_disp*4., spx_pred.float())
        student_outputs["init_disp_up"] = [init_disp_up]
        student_outputs["disp_preds"] = disp_preds

        return student_outputs

def init_dfmstereo_small(args=None, eval=True):
    checkpoint_path = Path(__file__).parent.parent / "pretrained" / f"dfmstereo_small_{args.dfmstereo_ckpt}.pth" if args is not None and hasattr(args, "dfmstereo_ckpt") else None

    config = Namespace(
        restore_ckpt = checkpoint_path,
        mixed_precision=args.mixed_precision if args and hasattr(args, "mixed_precision") else False,
        precision_dtype=args.precision_dtype if args and hasattr(args, "precision_dtype") else 'float32',
        train_iters=args.train_iters if args and hasattr(args, "train_iters") else 8,
        valid_iters=args.valid_iters if args and hasattr(args, "valid_iters") else 8,
        feature_backbone=args.feature_backbone if args and hasattr(args, "feature_backbone") else "edgenext",
        corr_groups=1,
        concat_groups=14,
        volume_dim=28,
        cv_agg_n_heads=args.cv_agg_n_heads if args and hasattr(args, "cv_agg_n_heads") else 4,
        max_disp=192,
        hidden_dim=64,
        context_dim=64,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        cor_planes=522,
        
        being_taught=args.being_taught if args and hasattr(args, "being_taught") else False,
        teacher=args.teacher if args and hasattr(args, "teacher") else "foundation_stereo",

        features=args.features if args and hasattr(args, "features") else False,
        factor_transfer=args.factor_transfer if args and hasattr(args, "factor_transfer") else False,
        cost_volume=args.cost_volume if args and hasattr(args, "cost_volume") else False,
        agg_cost_volume=args.agg_cost_volume if args and hasattr(args, "agg_cost_volume") else False,
        init_disp_logits=args.init_disp_logits if args and hasattr(args, "init_disp_logits") else False,
        refined_disp=args.refined_disp if args and hasattr(args, "refined_disp") else False,

        update_block_iters=args.update_block_iters if args and hasattr(args, "update_block_iters") else [],
    )
    
    # NOTE: Load checkpoint on cpu to avoid running out of memory.
    model = DFMStereo_Small(config) # init on cpu
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint Path not found: {checkpoint_path}")
        assert str(checkpoint_path).endswith(".pth")
        if args is not None and hasattr(args, "rank") and args.rank == 0:
            print(f"Loading DFMStereo-Small checkpoint '{args.dfmstereo_ckpt}' from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)  # load to cpu
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        if args is not None and hasattr(args, "rank") and args.rank == 0:
            print("No pre-trained weights for DFMStereo-Small loaded.")

    if eval:
        for param in model.parameters(): 
            param.requires_grad = False
        model = model.eval()
    else:
        for param in model.parameters(): 
            param.requires_grad = True
        model = model.train()

    if args is not None and hasattr(args, "device"):
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(DEVICE) # move to cuda if available

    return model
