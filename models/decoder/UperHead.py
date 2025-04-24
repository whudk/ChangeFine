

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


from lib.tools.module_helper import ModuleHelper

import math
from timm.models.layers import trunc_normal_,to_2tuple

import einops

import torch.distributed as dist


class UperHead(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(96, 192, 384, 786), fpn_dim=256,**kwargs):
        super(UperHead, self).__init__()

        bn_type = "torchsyncbn" if dist.is_initialized() else "torchbn"


        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, fpn_dim, kernel_size=1, bias=False),
                ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales)*fpn_dim, fpn_dim, kernel_size=3,stride = 1,padding = 1, bias=False),
            ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
        )


        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
                ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
                #conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            nn.Conv2d(len(fpn_inplanes) * fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

        self.norm = nn.LayerNorm(fc_dim * 4)

        self.norm1 = nn.LayerNorm(fc_dim)
        self.norm2 = nn.LayerNorm(fc_dim)
        self.norm3 = nn.LayerNorm(fc_dim)
        self.norm4 = nn.LayerNorm(fc_dim)


        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2),
            nn.GroupNorm(32, fc_dim),
            nn.GELU(),
            nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(2, 2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)


    def build_pyramid_features(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            b, hw, dim = x[0].size()
            #
            ori_h = ori_w = math.sqrt(hw)
            # if len(x) == 1:
            #     x = x[0]
            #     dim = dim // 4
            #     f1, f2, f3, f4 =  torch.split(x,dim ,dim=-1)
            # else:
            #     f1, f2, f3, f4 = x
            # f1 = self.norm1(f1).transpose(1, 2).reshape(-1, dim , int(ori_h), int(ori_w))
            # f2 = self.norm2(f2).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            # f3 = self.norm3(f3).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            # f4 = self.norm4(f4).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            #assert  x is isinstance(x, list)

            f = []
            for i, x_i in enumerate(x):
                _, _, dim = x_i.size()
                f.append(x_i.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w)))


            f1, f2, f3 , f4 = f
            #
            # f1 = f1.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            # f2 = f2.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            # f3 = f3.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
            # f4 = f4.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))



            f1 = self.up1(f1).contiguous()
            f2 = self.up2(f2).contiguous()
            f3 = self.up3(f3).contiguous()

            f4 = self.up4(f4).contiguous()
            return [f1, f2, f3, f4]
        else:
            b, hw, dim = x.size()
            #
            ori_h = ori_w = int(math.sqrt(hw))
            x = einops.rearrange(x, 'b  (h w) c -> b c h w', h = ori_h, w = ori_w)
            f1 = self.up1(x).contiguous()
            f2 = self.up2(x).contiguous()
            f3 = x
            f4 = self.up4(x).contiguous()
        return [f1, f2, f3, f4]
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




    def forward_fusion(self,conv_out, segSize = None):
        if len(conv_out[0].shape) == 3 and isinstance(conv_out,list):
            conv_out = self.build_pyramid_features(conv_out)

        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        #x = nn.functional.log_softmax(x, dim=1)

        return x,fusion_out
    def forward(self, conv_out, segSize=None, conv_last  = False):
        #if len(conv_out[0].shape) == 3 and isinstance(conv_out,list):
        conv_out = self.build_pyramid_features(conv_out)

        #self.build_pyramid_features(conv_out)
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        if conv_last:
            fusion_out = torch.cat(fusion_list, 1)
            x = self.conv_last(fusion_out)

            if self.use_softmax:  # is True during inference
                x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x
        else:
            return fusion_list
        #x = nn.functional.log_softmax(x, dim=1)



#UperHead for segmentor without backbone

class UperSegHead(UperHead):
    def __init__(self,
                 num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(96, 192, 384, 786), fpn_dim=256,
                 embed_dim = 512,
                 **kwargs
                 ):

        self.bn_type = "torchsyncbn" if dist.is_initialized() else "torchbn"#self.configer.get("network","bn_type")
        super(UperSegHead, self).__init__(
            num_class=num_class, fc_dim=fc_dim,
            use_softmax=use_softmax, pool_scales=pool_scales,
            fpn_inplanes=fpn_inplanes, fpn_dim=fpn_dim, bn_type= self.bn_type, **kwargs
        )
        self.embed_layer = nn.Sequential(
                nn.Conv2d(fc_dim*4, embed_dim,kernel_size=1,stride=1),
                ModuleHelper.BNReLU(embed_dim, bn_type= self.bn_type)
        )


        if self.configer.exists('contrast'):
            self.r = self.configer.get('contrast', 'memory_size')
            self.register_buffer("segment_queue", torch.randn(num_class, self.r, embed_dim))
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.register_buffer("segment_queue_ptr", torch.zeros(num_class, dtype=torch.long))
            self.register_buffer("pixel_queue", torch.randn(num_class, self.r, embed_dim))
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.register_buffer("pixel_queue_ptr", torch.zeros(num_class, dtype=torch.long))

        self.aux_head = nn.Conv2d(fc_dim * 4, num_class,kernel_size=1,stride=1)


    def forward(self, conv_out, targets = None,with_embed = False,is_eval = False):
        #if len(conv_out[0].shape) == 3 and isinstance(conv_out,list):
        conv_out = self.build_pyramid_features(conv_out)

        fusion_feats = []
        for idx in range(len(conv_out)):
            feats = F.interpolate(conv_out[idx], size=(conv_out[0].shape[2:]), mode='bilinear', align_corners=True)

            fusion_feats.append(feats)

        fusion_feats = torch.cat((fusion_feats), dim=1)


        aux_out = self.aux_head(fusion_feats)
        out = super(UperSegHead, self).forward(conv_out)



        embed_feats = None
        if with_embed is True:
            embed_feats = self.embed_layer(fusion_feats)


        if with_embed and is_eval is not True:
            return {'pred': out, 'pred_aux': aux_out, 'embed': embed_feats, 'key': embed_feats.detach()}
        else:
            return {'pred': out, 'pred_aux': aux_out, 'embed': embed_feats, 'key': embed_feats}


# class UperChgHead(UperSegHead):
#     def __init__(self,
#                  configer =None,
#                  num_class=150, fc_dim=4096,
#                  use_softmax=False, pool_scales=(1, 2, 3, 6),
#                  fpn_inplanes=(96, 192, 384, 786), fpn_dim=256,
#                  embed_dim = 512,
#                  **kwargs
#                  ):
#         super(UperChgHead, self).__init__(
#             configer = configer,
#             num_class=num_class, fc_dim=fc_dim,
#             use_softmax=use_softmax, pool_scales=pool_scales,
#             fpn_inplanes=fpn_inplanes, fpn_dim=fpn_dim ,**kwargs
#         )
#
#         self.num_classes = self.configer.get("data","num_classes")
#
#         self.fusion_type = self.configer.get("network","head")["fusion"]
#
#         self.mode = self.configer.get("train","mode")
#         if self.mode == 'shpcd':
#             self.shp_encoder = BackboneSelector(configer).get_backbone(self.configer.get('network','shp_encoder')["name"],
#                                                                        arch=self.configer.get("network", "shp_encoder")["name"],
#                                                                        net_params=self.configer.get("network", "shp_encoder")["params"],
#                                                                        )
#             out_dim = self.shp_encoder.embed_dim
#             self.embed_patch_shp = Mlp(out_dim, hidden_features=out_dim * 4, out_features=8192, drop=0.0)
#             self.embed_patch_img = Mlp(out_dim, hidden_features=out_dim * 4, out_features=8192, drop=0.0)
#             self.cross_fusion = CrossAttention_Block(embed_dim, num_heads=2, window_size=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
#                                  attn_drop=0., )
#
#
#
#         self.embed_layer =   TransformerMLP(fpn_dim * 4,expansion=embed_dim / (4 * fpn_dim),drop=0.0)
#         # nn.Sequential(
#         #         nn.Conv2d(fpn_dim*4, embed_dim,kernel_size=1,stride=1),
#         #         ModuleHelper.BNReLU(embed_dim, bn_type= self.bn_type)
#         # )
#
#
#
#
#
#         inchannels = fpn_inplanes
#
#         self.feats_interacts = []
#         self.fuse_convs = []
#
#
#
#         self.feats_interacts = nn.ModuleList([
#             CrossAttention_Block(dim, num_heads=2, window_size=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
#                                  attn_drop=0., )
#             for dim in inchannels])
#
#         self.fuse_convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(dim * 2, dim * 4, kernel_size=1),
#                 ModuleHelper.BNReLU(dim * 4, bn_type=self.bn_type),
#                 nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0),
#                 ModuleHelper.BNReLU(dim, bn_type=self.bn_type)
#             )
#             for dim in inchannels])
#
#
#
#         if self.configer.exists('contrast'):
#             self.r = self.configer.get('contrast', 'memory_size')
#             self.register_buffer("segment_queue", torch.randn(num_class, self.r, embed_dim))
#             self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
#             self.register_buffer("segment_queue_ptr", torch.zeros(num_class, dtype=torch.long))
#             self.register_buffer("pixel_queue", torch.randn(num_class, self.r, embed_dim))
#             self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
#             self.register_buffer("pixel_queue_ptr", torch.zeros(num_class, dtype=torch.long))
#
#         self.aux_head = nn.Conv2d(fc_dim * 4, num_class,kernel_size=1,stride=1)
#
#         #self.chg_head =  OCRHead(configer,in_channels=embed_dim,embed_dim = embed_dim // 2 , key_dim= 256 , num_classes= self.num_classes) # nn.Conv2d(embed_dim,  self.num_classes , kernel_size=1, stride=1, padding=0),
#         self.chg_head = UPerHead(
#             num_class = self.num_classes,fc_dim = 768,fpn_inplanes=[768,768,768,768],fpn_dim=256,bn_type=self.bn_type
#         )
#
#
#
#
#
#     def fusion_feats(self, x1, x2, fuse_layer = None, feat_layer_interact = None, fusion='addcd'):
#         outs = []
#         if fusion == 'addcd':
#             fuse = x1 - x2
#         elif fusion == 'cd':
#             fuse = torch.abs(x1 - x2)
#         elif fusion == 'concat':
#             fuse = torch.cat((x1, x2), dim=1)
#             assert fuse_layer is not None
#             fuse = fuse_layer(fuse)
#             #fuse = torch.split(fuse, fuse.shape[1] // 4, dim=1)
#         elif fusion == 'cross':
#             assert feat_layer_interact is not None
#             fuse = feat_layer_interact(x1, x2)
#         else:
#             raise ValueError("type of {} is not supported.".format(fusion))
#
#         # fuse = self.cam_pam(fuse)
#
#         return fuse
#
#     def forward_shp_img(self,shp_img,conv_out_img,with_embed = False,is_eval = False):
#
#         conv_out_shp = self.shp_encoder.get_intermediate_layers(shp_img,4)
#
#         x1 = conv_out_shp[-1] # B * n_patch * dim
#         x2 = conv_out_img[-1] # B * n_patch * dim
#         embed_patch_shp = self.embed_patch_shp(x1)
#         embed_patch_img = self.embed_patch_img(x2)
#
#
#
#
#
#
#         if len(conv_out_img[0].shape) == 3 and isinstance(conv_out_img,list):
#             conv_out_img = self.build_pyramid_features(conv_out_img)
#
#         if len(conv_out_shp[0].shape) == 3 and isinstance(conv_out_shp, list):
#             conv_out_shp = self.build_pyramid_features(conv_out_shp)
#
#         fusion_feats_img = []
#         for idx in range(len(conv_out_img)):
#             feats = F.interpolate(conv_out_img[idx], size=(conv_out_img[0].shape[2:]), mode='bilinear', align_corners=True)
#             fusion_feats_img.append(feats)
#
#         fusion_feats_shp = []
#         for idx in range(len(conv_out_shp)):
#             feats = F.interpolate(conv_out_shp[idx], size=(conv_out_shp[0].shape[2:]), mode='bilinear', align_corners=True)
#             fusion_feats_shp.append(feats)
#
#
#         b, n_patch, dim = x1.size()
#
#
#
#
#
#
#         fusion_feats = torch.cat((fusion_feats_img), dim=1)
#
#
#
#
#         aux_out = self.aux_head(fusion_feats)  #get aux pred of seg
#         out = super(UperSegHead, self).forward(conv_out_img) #get  pred of seg
#
#
#
#         fusions = []
#
#         for i, (feat_left,feat_right) in enumerate(zip(conv_out_img,conv_out_shp)):
#             feat = self.fusion_feats(feat_left,feat_right,fuse_layer=None,feat_layer_interact=self.feats_interacts[i],fusion = self.fusion_type)
#             fusions.append(feat)
#
#
#
#
#
#         chg_out = self.chg_head(fusions)#get  pred of change
#
#         if with_embed and is_eval is not True:
#             return {'pred': out, 'pred_aux': aux_out, 'pred_chg':chg_out, 'embed': embed_patch_img, 'embed_shp': embed_patch_shp}
#         else:
#             return {'pred': out, 'pred_aux': aux_out, 'pred_chg':chg_out , 'embed': embed_patch_img.detach(), 'embed_shp': embed_patch_shp.detach()}
#
#
#
#     def forward_img_img(self, conv_out_left, conv_out_right,  targets = None,with_embed = False,is_eval = False):
#         if len(conv_out_left[0].shape) == 3 and isinstance(conv_out_left,list):
#             conv_out_left = self.build_pyramid_features(conv_out_left)
#
#         if len(conv_out_right[0].shape) == 3 and isinstance(conv_out_right, list):
#             conv_out_right = self.build_pyramid_features(conv_out_right)
#
#         conv_out = []
#         for i, (x1, x2) in enumerate(zip(conv_out_left, conv_out_right)):
#
#             conv_out.append(
#                 self.fusion_feats(x1,x2,self.fuse_convs[i],self.feats_interacts[i],fusion = self.fusion_type)
#             )
#
#         fusion_feats = []
#         for idx in range(len(conv_out)):
#             feats = F.interpolate(conv_out[idx], size=(conv_out[0].shape[2:]), mode='bilinear', align_corners=True)
#
#             fusion_feats.append(feats)
#
#         fusion_feats = torch.cat((fusion_feats), dim=1)
#
#
#         aux_out = self.aux_head(fusion_feats)
#         out = super(UperSegHead, self).forward(conv_out)
#
#
#
#         embed_feats = None
#         if with_embed is True:
#             embed_feats = self.embed_layer(fusion_feats)
#
#
#         if with_embed and is_eval is not True:
#             return {'pred': out, 'pred_aux': aux_out, 'embed': embed_feats, 'key': embed_feats.detach()}
#         else:
#             return {'pred': out, 'pred_aux': aux_out, 'embed': embed_feats, 'key': embed_feats}
#
#     def forward(self,inputs,  targets = None,with_embed = False,is_eval = False):
#         if self.mode == 'cd':
#             conv_out_left,conv_out_right = inputs[0],inputs[1]
#             return self.forward_img_img(conv_out_left,conv_out_right,targets=targets,with_embed=with_embed,is_eval=is_eval)
#         else:
#             shp_img, conv_out_right = inputs[0], inputs[1]
#             return self.forward_shp_img(shp_img, conv_out_right, with_embed=with_embed, is_eval=is_eval)