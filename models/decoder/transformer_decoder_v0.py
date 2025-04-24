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
        text_embedding: Tensor,

    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          text_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)



        # Prepare queries
        queries = text_embedding
        keys = image_embedding

        # # Apply transformer blocks and final layernorm
        # for layer in self.layers:
        #     queries, keys = layer(
        #         queries=queries,
        #         keys=keys,
        #     )
        #     # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=text_embedding,
                key_pe=image_pe,
            )
        q = queries + text_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k,
                                                  v=keys)  # 点和key的注意力 k:含有坐标的特征， v:不含坐标的特征  q:含有坐标的点特征    from points to image
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
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: List,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
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

        self.num_mask_tokens = num_multimask_outputs
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
        self.H = self.W = img_size[0]

        self.corse_encoder = nn.Sequential(
            nn.Conv2d(self.num_mask_tokens, transformer_dim, kernel_size=1, stride=1),
            LayerNorm2d(transformer_dim),
            activation()
        )

        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         transformer_dim,
        #         transformer_dim,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(transformer_dim),
        #     nn.Conv2d(
        #         transformer_dim,
        #         transformer_dim,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(transformer_dim),
        # )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        corse_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder

          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        #image_embeddings = self.neck(image_embeddings)
        corse_embeddings = self.corse_encoder(corse_embeddings)
        #corse_embeddings = self.mask_tokens(corse_embeddings)
        b,c,h,w = image_embeddings.shape

        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=build_2d_sincos_position_embedding(h,w,embed_dim=self.transformer_dim),
            corse_embeddings= corse_embeddings,
            text_embeddings = text_embeddings
        )



        # Prepare output
        return masks



    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe:torch.tensor,
        corse_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        tokens = self.mask_tokens.weight
        tokens = tokens.unsqueeze(0).expand(corse_embeddings.size(0), -1, -1)
        tokens = torch.cat((tokens, text_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask

        if len(image_embeddings.shape) == 3:
            image_embeddings = einops.rearrange(image_embeddings, 'b (h w) c -> b c h w',h = int(math.sqrt(image_embeddings.shape[1])) )
        b, c, h, w = image_embeddings.shape

        image_embeddings = image_embeddings + corse_embeddings

        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0).to(image_embeddings.device)

        # Run the transformer
        hs, src = self.transformer(image_embeddings,pos_src, tokens )

        mask_tokens_out = hs

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        masks = F.interpolate(masks, (self.H, self.W), mode='bilinear', align_corners=True)

        # Generate mask quality predictions


        return masks


class MultiMaskDecoder(MaskDecoder):
    def __init__(self,
                 *,
                 img_size = [256],
                 text_dim= 512,
                 transformer_dim: int,
                 transformer: nn.Module,
                 num_multimask_outputs: int = 2,
                 activation: Type[nn.Module] = nn.GELU,
                 n_feats=4,
                 num_class = 2,
                 **kwargs
                 ):
        super(MultiMaskDecoder, self).__init__(
            img_size=img_size,
            transformer_dim  = transformer_dim,
            transformer = transformer,
            num_multimask_outputs = num_multimask_outputs,
            activation = activation,
            **kwargs
        )
        self.conv_out = nn.Conv2d(n_feats * num_class, num_class,kernel_size=1, stride=1)

        self.norm_t = nn.LayerNorm(transformer_dim)
    def forward(self, x_list, corse_pred,t_embedding):
        masks = []
        t_embedding = self.norm_t(t_embedding)
        corse_pred = torch.softmax(corse_pred, dim=1)
       # corse_pred=torch.zeros_like(x_list[0],device=t_embedding.device)
        for x in x_list:
            masks.append(super(MultiMaskDecoder, self).forward(x, corse_pred, t_embedding))
        masks = torch.cat(masks,dim=1)
        masks = self.conv_out(masks)
        return masks


if __name__ == '__main__':
    import torch

    # Define some hyperparameters
    B = 2  # Batch size
    embedding_dim = 512  # Example embedding dimension
    num_heads = 8  # Example number of attention heads
    mlp_dim = 2048  # Example MLP dimension
    depth = 6  # Number of transformer layers
    activation = torch.nn.ReLU  # Activation function to use
    attention_downsample_rate = 2  # Attention downsampling rate
    num_multimask_outputs = 2  # Number of mask outputs
    text_length = 2  # Number of tokens in text input

    # Create random image and text embeddings
    image_embedding = torch.randn(B, embedding_dim, 64, 64)  # Example image embedding of shape B x C x H x W
    text_embedding = torch.randn(B, text_length, embedding_dim)  # Example text embedding of shape B x N x C

    # Initialize the TwoWayTransformer model
    transformer_model = TwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=activation,
        attention_downsample_rate=attention_downsample_rate,
    )

    # Initialize the MaskDecoder model
    mask_decoder = MaskDecoder(
        transformer_dim=embedding_dim,
        transformer=transformer_model,
        num_multimask_outputs=num_multimask_outputs,
        activation=activation,
    )

    # Forward pass through the MaskDecoder
    multimask_output = True  # Whether to return multiple masks
    masks = mask_decoder(image_embeddings=image_embedding, text_embeddings=text_embedding,
                         multimask_output=multimask_output)

    # Print the output shapes
    print("Output shape of masks:", masks.shape)  # Expected shape: (B, num_masks, H, W)
