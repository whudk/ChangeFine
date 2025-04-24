import math
from typing import Any, Dict, List, Tuple

import einops
import torch.nn as nn
import torch
from models.clip.clip import tokenize
from models.clip import clip
from models.clip.model import CLIPVisionTransformer, ContextDecoder, CLIPTextContextEncoder,cross_clip_sam,cross_attention
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
import  numpy as np
_tokenizer = _Tokenizer()


from models.attention import Cross_Modal_Attention

from Inference.hdrpylc.hdrpylc import checkLib
from models.decoder.transformer_decoder import MaskDecoder,TwoWayTransformer,MultiMaskDecoder,ChangeDecoder
from  models.attention import Mlp


from models.decoder.BuildHead import BuildHead

from inspect import isfunction

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final

        self.dtype = clip_model.dtype

        self.text_projection = clip_model.text_projection



    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        k_x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)

        return x , k_x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.get("COOP","N_CTX")#cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.get("COOP","CTX_INIT")#cfg.TRAINER.COOP.CTX_INIT fix prompt
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.get("COOP","input_size")#cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.get("COOP","CSC"):
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("-", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        #self.time_embeddings = nn.Parameter(torch.randn(2, ctx_dim))  # 两个时间步的嵌入
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx#context
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.get("COOP","CLASS_TOKEN_POSITION")#cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self,  timesteps = 0):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        #self.time_embeddings = nn.Parameter(torch.randn(2, prompt_dim))  # 两个时间步的嵌入

        #prompts = self.time_embeddings[timesteps] + prompts
        return prompts

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)  # squeeze
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()  # excitation
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)         # squeeze to [B, C]
#         y = self.fc(y).view(b, c, 1, 1)     # [B, C, 1, 1]
#         return x * y.expand_as(x)
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    # def forward(self,x):
    #
    #     return self.conv(x)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.shortcut(x)
        return   self.conv(x) +  residual

class MoEConvFusion4Experts(nn.Module):
    def __init__(self, in_channels,  out_dim = 256,use_norm=True, drop = 0.):
        super().__init__()
        self.in_channels = in_channels  # C
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(in_channels) if use_norm else nn.Identity()

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 4, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 4)  # 4 experts
        )

        # self.out_proj = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.ReLU()
        # )
        self.out_proj = ResidualConvBlock(in_channels = in_channels, out_channels=out_dim)
        #self.out_proj = Mlp(in_features=in_channels, hidden_features= 2 * in_channels, out_features=out_dim,drop=drop)
    def forward(self, x):
        assert  isinstance(x, list) or isinstance(x. tuple)

        # Split 4 experts: [B, C, H, W]
        x1, x2, x3, x4 = x


        if len(x1.shape) == 4:

            B,C ,H,W = x1.shape

            x1 = einops.rearrange(x1,'b c h w -> b (h w) c')
            x2 = einops.rearrange(x2, 'b c h w -> b (h w) c')
            x3 = einops.rearrange(x3, 'b c h w -> b (h w) c')
            x4 = einops.rearrange(x4, 'b c h w -> b (h w) c')
        else:
            H = W = int(math.sqrt(x1.shape[1]))

        x1, x2, x3, x4 = map(self.norm, (x1, x2, x3, x4))

        # Gating input: [B, N, 4C]
        gate_input = torch.cat([x1, x2, x3, x4], dim=-1)
        gate_weight = F.softmax(self.gate(gate_input), dim=-1)  # [B, N, 4]

        # MoE fusion
        experts = torch.stack([x1, x2, x3, x4], dim=-2)  # [B, N, 4, C]
        fused = torch.sum(gate_weight.unsqueeze(-1) * experts, dim=-2)  # [B, N, C]




        fused = einops.rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)

        # Output projection
        fused = self.out_proj(fused)
        if len(fused.shape) == 3:
            fused = einops.rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)


        return fused
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats = 64, scale = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class ChangeFine(nn.Module):
    def __init__(self,
                 configer,
                 clip_model,
                 sam_model,
                 context_length,
                 class_names,
                 sam_dim = 768,
                 clip_dim = 768,
                 text_dim = 512,
                 width = 512,
                 feats_exchange="CCMAT",
                 feats_fusion="TBAM",
                 n_cross_head=16,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 mask_threshold = 0.0,
                 prompt = None,
                 decoder_width = 512,
                 n_pts = 5000,
                 n_layer = 6,
                 **kwargs):
        super().__init__()
        self.dtype = clip_model.dtype
        self.mask_threshold = mask_threshold
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = sam_model.image_encoder
        for param in self.parameters():
            param.requires_grad = False

        self.pe_encoder = PositionEmbeddingRandom(decoder_width // 2)
        self.out_cls = configer.get("data", "num_classes")
        self.len_classes = len(class_names)
        self.width = width
        self.clip_image_encoder = CLIPVisionTransformer(input_resolution=configer.get("data", "input_size"),
                                                        pretrained=clip_model, patch_size=16, get_embeddings=True)
        self.prompt_learner = PromptLearner(configer, class_names, clip_model)


        self.norm_sam = nn.LayerNorm(sam_dim)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        self.prompt = prompt
        assert self.prompt in ["clip", "sam", "clipsam", "samclip"]

        self.context_visual_attn = cross_attention(
            m1_dim=text_dim,
            m2_dim=text_dim,
            out_dim=width,
            transformer_width=width,
            transformer_heads=8,
            transformer_layers=1
        )
        #self.gamma_coarse = nn.Parameter(torch.zeros(1))
        self.cross_clip_sam = cross_clip_sam(clip_dim = text_dim,sam_dim = sam_dim,transformer_width=width,transformer_heads=8,transformer_layers=n_layer,out_dim=width)

        if self.prompt == 'clip':  # clip with sam
            interaction_dim = clip_dim
        elif self.prompt == "sam":
            interaction_dim = sam_dim
        elif self.prompt == 'clipsam':
            interaction_dim = sam_dim
        elif self.prompt == 'samclip':
            interaction_dim = clip_dim
        else:
            raise ValueError
        self.num_classes = len(class_names)
        self.embed_img_t  = Mlp(in_features=width + self.num_classes, hidden_features = 4 * (width + self.num_classes) ,out_features=interaction_dim, drop = proj_drop)


        self.fusion_score = Mlp(in_features = 2 * self.num_classes, hidden_features = 4 * (2 * self.num_classes) ,out_features=self.out_cls, drop = proj_drop)

        self.interacts = nn.ModuleList(
            [
                    Cross_Modal_Attention(interaction_dim, width, num_heads=n_cross_head, depth=1,
                                          window_size=1,
                                          mlp_ratio=4., qkv_bias=False,
                                          qk_scale=None, drop=proj_drop,
                                          attn_drop=attn_drop, feats_fusion=feats_fusion,
                                          feats_exchange=feats_exchange)

                    for i in range(4)
            ]
        )



        # Register buffers to store positive and negative features
        self.register_buffer("mem_queue", torch.randn((self.out_cls, n_pts, decoder_width)))
        self.segment_queue = nn.functional.normalize(self.mem_queue, p=2, dim=2)
        self.register_buffer("mem_queue_ptr", torch.zeros(self.out_cls, dtype=torch.long))

        neck_dict = configer.get("network", "neck")
        neck_dict["params"]["num_class"] = self.out_cls
        self.neck = BuildHead(configer).build_head(name=neck_dict["name"], **neck_dict["params"])

        self.MoE = False
        if configer.get("network", "params")["feats_fusion"] =='SCAM':
            self.embed_feats = MoEConvFusion4Experts(
             width, decoder_width)
            self.MoE = True
        else:
            self.embed_feats = nn.Sequential(
                nn.Conv2d(width * 4, decoder_width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(decoder_width),
                nn.LeakyReLU(0.01),
                nn.Conv2d(decoder_width, decoder_width, kernel_size=3, stride=1, padding=1),
            )


        self.aux_head = nn.Sequential(
            nn.Conv2d(decoder_width , self.out_cls, kernel_size=1, stride=1, padding=0)
        )

        self.embed_text = Mlp(in_features=width, hidden_features=width*2,
                             out_features=decoder_width, drop=proj_drop)

        self.decoder = MultiMaskDecoder(
            transformer_dim = decoder_width ,
            transformer  = TwoWayTransformer(
                depth=2,
                embedding_dim=decoder_width,
                num_heads=8,
                mlp_dim=1024
            ),
            num_multimask_outputs = self.out_cls ,
            activation=nn.ReLU,
            n_feats = 1,
            num_class= self.out_cls
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的加权系数

    def  fusion_clip_to_sam(self, t_embedding, clip_feats, sam_feats, fusion = "sam_clip"):


        if fusion == 'clip':
            return clip_feats[:4], None, None
        elif fusion == 'sam':
            return sam_feats, None, None
        sam_feats = self.norm_sam(sam_feats)
        global_feat, visual_context = clip_feats[4]
        B, C, H, W = visual_context.shape
        clip_feats = torch.cat([global_feat.reshape(B, C, 1), visual_context.reshape(B, C, H * W)], dim=2).permute(
            0, 2, 1)  # B, N, C

        # (B, K, C)

        text_embeddings = t_embedding.expand(clip_feats.shape[0], -1, -1).float()
        # input text_embeddings: B X K X C     visual_context: B X N X C
        text_diff = self.context_visual_attn(text_embeddings, clip_feats)  # attention(q,[z_hat,z])

        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = clip_feats
        text = F.normalize(text_embeddings, dim=2, p=2)


        if fusion == 'clipsam':
            visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_feats,"clip")
        else:
            visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_feats, "sam")


        score_map = visual_embeddings @ text.transpose(-2,-1)  # torch.einsum('bcn,bkc->bkn', visual_embeddings, text)  # z @ t
        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        # if prompt_sam is not False:
        #     # 修改shape
        #     return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        # else:
        if fusion == 'clipsam':
            return  visual_embeddings, score_map, text_embeddings
        else:
            return visual_embeddings[:, 1:, :], score_map[:, 1:, :], text_embeddings

    def SCFE(self, x, t_embedding):



        with torch.no_grad():
            _, sam_feats = self.image_encoder(x)
            clip_feats = self.clip_image_encoder(x)

        if self.prompt == 'clip':
            return  list(clip_feats[:4]), None, None
        elif self.prompt == 'sam':
            return sam_feats, None, None
        elif self.prompt=='clipsam':
            x_fuse = sam_feats

            sam_last = einops.rearrange(sam_feats[-1], "b c h w -> b (h w) c")

            x_o, score_map, t_embedding = self.fusion_clip_to_sam(t_embedding, clip_feats, sam_last, self.prompt)
            x_o = einops.rearrange(x_o, "b (h w) c -> b c h w", h = int(math.sqrt(x_o.shape[1])))

            x_fuse[-1] = x_o + x_fuse[-1]
            return x_fuse, score_map, t_embedding
        elif self.prompt == 'samclip':
            x_fuse = list(clip_feats[:4])
            sam_last = einops.rearrange(sam_feats[-1], "b c h w -> b (h w) c")
            x_o, score_map, t_embedding = self.fusion_clip_to_sam(t_embedding, clip_feats, sam_last, self.prompt)
            x_o = einops.rearrange(x_o, "b (h w) c -> b c h w", h=int(math.sqrt(x_o.shape[1])))

            x_fuse[-1] = x_o + x_fuse[-1]
            return x_fuse, score_map, t_embedding
        else:
            raise ValueError(f'Unsupported prompt {self.prompt}')

    def sample_topk_foreground_points(self, coarse_pred, k = 10, class_idx = None):
        """
        从 coarse_pred 中采样每张图前景 Top-K 点（最大类别或指定类）作为提示。

        Args:
            coarse_pred: [B, C, H, W] — 每类的概率图
            k: int — 每张图采样 K 个提示点
            class_idx: int or None — 若指定类，则只在该类中采样；否则用每图最大类

        Returns:
            point_coords: [B, K, 2] — 归一化坐标
            point_labels: [B, K] — label=1 表示前景
        """

        B, C, H, W = coarse_pred.shape
        coarse_pred_softmax = F.softmax(coarse_pred, dim=1)
        point_coords = []
        point_labels = []

        for b in range(B):
            probs = coarse_pred_softmax[b]  # [C, H, W]

            if class_idx is not None:
                class_map = probs[class_idx]
            else:
                class_map, _ = probs.max(dim=0)  # [H, W]

            # flatten to [H*W]
            class_map_flat = class_map.flatten()  # [H*W]
            topk_vals, topk_idxs = torch.topk(class_map_flat, k)

            y = topk_idxs // W
            x = topk_idxs % W

            x_norm = x.float() / W
            y_norm = y.float() / H

            coords = torch.stack([x_norm, y_norm], dim=-1)  # [K, 2]
            labels = torch.ones(k, dtype=torch.int, device=coarse_pred.device)  # [K]

            point_coords.append(coords.unsqueeze(0))  # [1, K, 2]
            point_labels.append(labels.unsqueeze(0))  # [1, K]

        point_coords = torch.cat(point_coords, dim=0)  # [B, K, 2]
        point_labels = torch.cat(point_labels, dim=0)  # [B, K]

        return point_coords, point_labels

    def forward(self, x1, x2, targets = None,batched_input = None, multimask_output = False, with_amp = False ):
        #encoder text
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts #K x C

        with torch.no_grad():
            _, t_embedding = self.text_encoder(prompts, tokenized_prompts)  # K X C
        x1_orig, score_x1, t1_embedding = self.SCFE(x1, t_embedding)
        x2_orig, score_x2, t2_embedding = self.SCFE(x2, t_embedding)
        x_fuses = []
        for i, (f1, f2) in enumerate(zip(x1_orig, x2_orig)):
            f1 = einops.rearrange(f1, "b c h w -> b (h w) c")
            f2 = einops.rearrange(f2, "b c h w -> b (h w) c")
            x_fuses.append(self.interacts[i](f1, f2))
        if score_x1 is not None and score_x2 is not None:
            score_fusion = self.fusion_score(torch.cat((score_x1, score_x2), dim = -1))
        else:
            b, n, c = x_fuses[0].shape
            score_fusion = torch.ones((b,n,self.out_cls),device=x_fuses[0].device)




        x_fuses[-2] = torch.cat((x_fuses[-2], score_fusion), dim=-1)
        x_fuses = self.neck(x_fuses)







        if self.MoE:
            dense_embed = self.embed_feats(x_fuses)
        else:
            dense_embed = self.embed_feats(torch.cat((x_fuses),dim = 1))

        if t1_embedding is not None and t2_embedding is not None:
            sim_matrix = torch.einsum('bid,bjd->bij', t1_embedding, t2_embedding)  # [B, L1, L2]
            attn = 1.0 - torch.softmax(sim_matrix, dim=-1)  # 不相似区域更高权重

            # 用注意力对 embedding 做加权
            t1_diff = torch.bmm(attn, t2_embedding)  # [B, L1, D]
            t2_diff = torch.bmm(attn.transpose(1, 2), t1_embedding)  # [B, L2, D]

            t_embedding = t1_diff + t2_diff  # 聚合后的差异信息
            t_embedding =self.embed_text(t_embedding)
        else:
            b, c, h, w = dense_embed.shape
            t_embedding = torch.ones((b,self.len_classes,self.width),device=dense_embed.device )

        coarse_pred = self.aux_head(dense_embed)
        sparse_prompt =  t_embedding#torch.empty((dense_embed.shape[0], 0, dense_embed.shape[1]), device=dense_embed.device)
        #sparse_prompt = torch.cat((sparse_embed, score_fusion.permute(0,2,1)), dim = 1)
        dense_pe = self.pe_encoder( dense_embed.shape[-2:]).to(dense_embed.device).unsqueeze(0)
        dense_prompt = coarse_pred
        if self.training:
            final_pred = self.decoder([dense_embed], dense_prompt, dense_pe, sparse_prompt, multimask_output = True)
            coarse_up = F.interpolate(coarse_pred, size=final_pred.shape[2:], mode='bilinear', align_corners=False)
            final_pred = self.alpha * final_pred + (1 - self.alpha) * coarse_up
        else:
            for i in range(3):
                dense_prompt = F.interpolate(dense_prompt, size=dense_embed.shape[2:], mode='bilinear', align_corners=False)
                final_pred = self.decoder([dense_embed], dense_prompt, dense_pe, sparse_prompt, multimask_output=True)
                coarse_up = F.interpolate(coarse_pred, size=final_pred.shape[2:], mode='bilinear', align_corners=False)
                dense_prompt = self.alpha * final_pred + (1 - self.alpha) * coarse_up
            final_pred =  dense_prompt



        return {"pred": final_pred, "dense_embed": dense_embed, "pred_aux": coarse_pred}





if __name__ == "__main__":
    pass