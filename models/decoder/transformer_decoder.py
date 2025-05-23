# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from typing import Tuple, Type,List
import einops
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.):
    """
    生成 2D sin-cos 位置编码，适配 ViT-style 的 image_pe。
    Args:
        h (int): 高度（patch grid H，比如 16）
        w (int): 宽度（patch grid W，比如 16）
        embed_dim (int): 通道数（对应 transformer_dim）
        temperature (float): 正弦编码的温度系数
    Returns:
        pos_embed: (1, embed_dim, h, w)
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij"
    )  # 形状：[H, W]

    assert embed_dim % 4 == 0, "embed_dim 必须能被 4 整除"

    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1. / (temperature ** omega)  # [C/4]

    # expand to [H, W, C/4]
    y = grid_y.unsqueeze(-1) * omega  # [H, W, C/4]
    x = grid_x.unsqueeze(-1) * omega

    pos_y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)  # [H, W, C/2]
    pos_x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, C]
    pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    return pos  # 可以直接作为 image_pe 使用


class MaskDecoderFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.final = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x  # [B, num_class, H, W]

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.dense_conv = nn.Sequential(
            nn.Conv2d(1, transformer_dim , kernel_size=1, padding = 0),
            LayerNorm2d(transformer_dim),
            activation(),
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        dense_prompt_embeddings = self.dense_conv(dense_prompt_embeddings)
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
             mask_slice = slice(1, None)
        else:
             mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        #src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

def dilate_mask(mask: torch.Tensor, kernel_size=5):
    """
    mask: [B, 1, H, W] float or binary
    returns: [B, 1, H, W] dilated mask
    """
    B, _, H, W = mask.shape
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=kernel_size // 2)
    return (dilated > 0).float()  # or > threshold


def erode_mask(mask: torch.Tensor, kernel_size=3):
    """
    mask: [B, 1, H, W] float or binary
    returns: [B, 1, H, W] eroded mask
    """
    B, _, H, W = mask.shape
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Perform erosion with a convolution operation
    eroded = F.conv2d(mask, kernel, padding=kernel_size // 2)

    # Return the eroded mask (binary)
    return (
                eroded == kernel_size * kernel_size).float()  # Only keep the pixels that are fully eroded (all 1s in the kernel)
class MultiMaskDecoder(MaskDecoder):
    def __init__(self,
                 *,
                 transformer_dim: int,
                 transformer: nn.Module,
                 num_multimask_outputs: int = 2,
                 activation: Type[nn.Module] = nn.GELU,
                 n_feats=4,
                 num_class = 2,
                 **kwargs
                 ):
        super(MultiMaskDecoder, self).__init__(
            transformer_dim  = transformer_dim,
            transformer = transformer,
            num_multimask_outputs = num_multimask_outputs,
            activation = activation,
            **kwargs
        )

        # self.mask_weight = nn.Parameter(torch.ones(num_multimask_outputs), requires_grad=True)
        #
        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(num_multimask_outputs, num_class, kernel_size=1, stride=1),
        # )

        self.mask_weight = nn.Parameter(torch.ones(num_multimask_outputs), requires_grad=True)

        self.conv_out = nn.Conv2d(num_multimask_outputs, num_class, kernel_size=1, stride=1)



        self.norm_t = nn.LayerNorm(transformer_dim)

    def forward(self, image_embeddings, dense_embed, dense_pe, sparse_embed, multimask_output=True):
        B, C, H, W = dense_embed.shape
        sparse_embed = self.norm_t(sparse_embed)

        # Step 1: Coarse → Binary → Dilate
        with torch.no_grad():
            coarse_mask = torch.argmax(dense_embed, dim=1, keepdim=True).float()  # [B,1,H,W]
            #coarse_mask = dilate_mask(coarse_mask, kernel_size=3)
            change_mask = (coarse_mask.flatten(1).sum(dim=1) > 0)  # [B]

        #final_mask = torch.zeros((B, self.conv_out.out_channels, H * 4, W * 4), device=dense_embed.device).float()
        final_mask = F.interpolate(dense_embed, size=(H * 4,  W * 4), mode='bilinear', align_corners=False).float()

        if change_mask.any():
            idx = change_mask.nonzero(as_tuple=True)[0]

            x_sub = image_embeddings[0][idx]
            pe_sub = dense_pe
            s_sub = sparse_embed[idx]
            d_sub = coarse_mask[idx]

            # 一次性 forward
            masks_sub, _ = super(MultiMaskDecoder, self).forward(
                x_sub, pe_sub, s_sub, d_sub, multimask_output=multimask_output
            )
            masks_sub = masks_sub * self.mask_weight.view(1, -1, 1, 1)
            output_sub = self.conv_out(masks_sub)  # [B', num_class, H, W]

            # 写回原 batch 中对应位置
            final_mask[idx] = output_sub.float()

        return final_mask

    # corse_pred=torch.zeros_like(x_list[0],device=t_embedding.device)
       #  for x in x_list:
       #      masks.append(super(MultiMaskDecoder, self).forward(x, dense_pe,sparse_embed,dense_embed,  False))
       #  masks = torch.cat(masks,dim=1)
       #  masks = self.conv_out(masks)
       #  return masks

from models.sam.modeling.prompt_encoder import PositionEmbeddingRandom


from timm.models.layers import trunc_normal_,to_2tuple

class ChangeDecoder(MaskDecoder):
    def __init__(self,
                 *,
                 transformer_dim: int,
                 transformer: nn.Module,
                 num_multimask_outputs: int = 2,
                 activation: Type[nn.Module] = nn.GELU,
                 n_feats=4,
                 fpn_planes = [512,512,512,512],
                 num_class=2,
                 **kwargs):
        super(ChangeDecoder, self).__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            **kwargs
        )

        # Define mask weights for each output mask
        self.mask_weight = nn.Parameter(torch.ones(num_multimask_outputs), requires_grad=True)

        # Define output layers for each level's predictions
        self.conv_out = nn.ModuleList([
            nn.Conv2d(num_multimask_outputs, num_class, kernel_size=1, stride=1) for _ in range(4)
        ])

        # Define convolution layers for coarse predictions at each level
        self.conv_pred = nn.ModuleList([
            nn.Conv2d(fpn_planes[3] + num_multimask_outputs, num_multimask_outputs, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_planes[2] + num_multimask_outputs, num_multimask_outputs, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_planes[1] + num_multimask_outputs, num_multimask_outputs, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_planes[0] , num_multimask_outputs, kernel_size=3, stride=1, padding=1),
        ])

        # Define convolution layers for upsampling (decoder layers)
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2) for _ in range(4)
        ])

        # Define positional encoding (PE) encoder
        self.pe_encoders = nn.ModuleList([
           PositionEmbeddingRandom(fpn_planes[i] // 2) for i in range(4)
        ])

        # Layer normalization for sparse embeddings
        self.norm_t = nn.LayerNorm(transformer_dim)

        # Define the mask decoders for each level
        self.masks_decoder = nn.ModuleList([
            MaskDecoder(transformer_dim=fpn_planes[i], transformer=transformer, num_multimask_outputs=num_multimask_outputs, activation=activation)
            for i in range(4)
        ])

        # Learnable weights for the level-wise predictions (Weighted Averaging)
        self.level_weights = nn.Parameter(torch.ones(4), requires_grad=True)

        self.norm1 = nn.LayerNorm(fpn_planes[0])
        self.norm2 = nn.LayerNorm(fpn_planes[1])
        self.norm3 = nn.LayerNorm(fpn_planes[2])
        self.norm4 = nn.LayerNorm(fpn_planes[3])

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(transformer_dim, transformer_dim//2, 2, 2),
            nn.GroupNorm(32, transformer_dim //2),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim//2, fpn_planes[0], 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(transformer_dim, fpn_planes[1], 2, 2)
        self.up3 = nn.Identity()
        self.up4 =nn.Conv2d(transformer_dim, fpn_planes[3], 2, 2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def build_pyramid_features(self, x):
        assert isinstance(x, tuple) or isinstance(x, list)
        f1, f2, f3, f4 = x
        b,hw,dim = f1.shape
        h = w = int(math.sqrt(hw))
        # b, dim, h,w = f1.size()
        # f1 = einops.rearrange(f1, 'b c h w -> b (h w) c')
        # f2 = einops.rearrange(f2, 'b c h w -> b (h w) c')
        # f3 = einops.rearrange(f3, 'b c h w -> b (h w) c')
        # f4 = einops.rearrange(f4, 'b c h w -> b (h w) c')
        f1 = self.norm1(f1).transpose(1, 2).reshape(-1, dim, h, w)
        f2 = self.norm2(f2).transpose(1, 2).reshape(-1, dim, h, w)
        f3 = self.norm3(f3).transpose(1, 2).reshape(-1, dim, h, w)
        f4 = self.norm4(f4).transpose(1, 2).reshape(-1, dim, h, w)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()

        f4 = self.up4(f4).contiguous()
        return [f1, f2, f3, f4]

    def process_level(self, x, level, sparse_embed, dense_embed, multimask_output=True, skip_connection=None):
        """ Helper function to process each level with skip connection """
        # Coarse prediction
        b, c ,h, w = x.shape

        if skip_connection is not None:
            skip_connection = F.interpolate(skip_connection, size=(h, w), mode='bilinear', align_corners=True)
            coarse_pred = torch.argmax(self.conv_pred[level](torch.cat([x, skip_connection], dim=1)), dim=1, keepdim = True).float()
        else:
            dense_embed = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            coarse_pred = torch.argmax(torch.softmax(dense_embed, dim=1),dim = 1,keepdim=True).float()
        #coarse_pred = torch.randn((b,1,h,w))
        # Adjust positional encoding (dense_pe) according to the size of the feature map at each level
        dense_pe = self.pe_encoders[level](x.shape[-2:]).to(dense_embed.device).unsqueeze(0)  # Adjust for current feature map size

        # Apply decoder
        masks, _ = self.masks_decoder[level](x, dense_pe, sparse_embed, coarse_pred, multimask_output)


        # Final output layer
        return self.conv_out[level](masks)

    def forward(self, image_embeddings, dense_embed, sparse_embed, multimask_output=True):
        B, C, H, W = dense_embed.shape

        # Normalize sparse embeddings
        sparse_embed = self.norm_t(sparse_embed)

        pyramid_features = self.build_pyramid_features(image_embeddings)

        # Unpack the pyramid features for each level
        x0, x1, x2, x3 = pyramid_features

        # Process each level using the helper function and add skip connections
        skip_connections = [None] * 4
        predictions = []

        # Level 4 (Bottom-most level)
        x3_pred = self.process_level(x3, 3, sparse_embed, dense_embed, multimask_output, skip_connection=None)
        predictions.append(x3_pred)

        # Level 3
        x2 = self.upconv[3](x3) + x2  # Skip connection from x4_pred
        x2_pred = self.process_level(x2, 2, sparse_embed, dense_embed, multimask_output, skip_connection=x3_pred)
        predictions.append(x2_pred)

        # Level 2
        x1 = self.upconv[2](x2) + x1  # Skip connection from x3_pred
        x1_pred = self.process_level(x1, 1, sparse_embed, dense_embed, multimask_output, skip_connection=x2_pred)
        predictions.append(x1_pred)

        # Level 1
        x0 = self.upconv[1](x1) + x0  # Skip connection from x2_pred
        x0_pred = self.process_level(x0, 0, sparse_embed, dense_embed, multimask_output, skip_connection=x1_pred)
        predictions.append(x0_pred)

        # Level 0 (Top-most level)

        return x0_pred


if __name__ == '__main__':
    import torch

    # Define some hyperparameters
    B = 1  # Batch size
    embedding_dim = 512  # Example embedding dimension
    num_heads = 8  # Example number of attention heads
    mlp_dim = 2048  # Example MLP dimension
    depth = 6  # Number of transformer layers
    activation = torch.nn.ReLU  # Activation function to use
    attention_downsample_rate = 2  # Attention downsampling rate
    num_multimask_outputs = 2  # Number of mask outputs
    text_length = 2  # Number of tokens in text input

    # Create random image and text embeddings
    image_embedding = 4*[torch.randn(B, embedding_dim, 16, 16)]  # Example image embedding of shape B x C x H x W
    text_embedding = torch.randn(B, text_length, embedding_dim)  # Example text embedding of shape B x N x C
    dense_embed = torch.randn(B, embedding_dim, 16, 16)
    decoder = ChangeDecoder(
        transformer_dim=embedding_dim,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=1024
        ),
        num_multimask_outputs=2,
        activation=nn.ReLU,
        n_feats=1,
        num_class=2
    )

    masks = decoder(image_embedding, dense_embed, text_embedding, multimask_output=True)

    # Print the output shapes
    print("Output shape of masks:", masks.shape)  # Expected shape: (B, num_masks, H, W)
