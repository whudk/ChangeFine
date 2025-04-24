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

_tokenizer = _Tokenizer()


from models.attention import Cross_Modal_Attention

from Inference.hdrpylc.hdrpylc import checkLib
from models.decoder.transformer_decoder import MaskDecoder,TwoWayTransformer,MultiMaskDecoder
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





class SamClipCD(nn.Module):
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
                 feats_fusion="TBAM",  # "CONCAT", "DIFF"
                 n_cross_head=16,
                 n_cross_layer=1,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 mask_threshold = 0.0,
                 prompt = None,
                 n_pts = 5000,
                 **kwargs):
        super().__init__()
        self.dtype = clip_model.dtype   #模型的数据类型
        self.mask_threshold = mask_threshold    #阈值
        self.text_encoder = TextEncoder(clip_model) #文本编码器
        self.image_encoder = sam_model.image_encoder    #图像编码器
        #冻结所有参数，这些参数不会更新
        for param in self.parameters():
            param.requires_grad = False
        self.norm_sam = nn.LayerNorm(sam_dim)
        self.norm_clip = nn.LayerNorm(clip_dim)
        self.embed_t0 = nn.Linear(text_dim, clip_dim)
        #self.embed_t1 = nn.Linear(text_dim, clip_dim)
        self.gamma = nn.Parameter(torch.ones(clip_dim) * 1e-4)
        self.context_visual_attn = cross_attention(
            m1_dim=clip_dim,
            m2_dim=clip_dim,
            out_dim=width,
            transformer_width=width,
            transformer_heads=8,
            transformer_layers=1
        )

        #如果prompt为sam，则prompt_sam为true,否则为None
        self.prompt = prompt
        assert self.prompt in ["clip","sam", "clipsam"]

        #初始化图像编码器，输入大小为256，patch_size为图像patch,即小区域提取特征
        self.clip_image_encoder = CLIPVisionTransformer(input_resolution=256 ,pretrained=clip_model, patch_size= 16, get_embeddings=False)


        #prompt_learner，学习提示词的模块，将class_names转化为适合CLIP的提示,一般为数值序列
        self.prompt_learner = PromptLearner(configer, class_names, clip_model)
        #self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        #texts通过tokenize函数将class_names转化为文本，num_classes为类别数量
        self.context_feature = 'attention'
        self.context_length = context_length
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        #加载sam_model时为256，不加载时为768
        self.cross_clip_sam = cross_clip_sam(
                                             clip_dim = clip_dim,
                                             sam_dim = sam_dim,
                                             transformer_width=width,
                                             transformer_heads=8,
                                             transformer_layers=6,
                                             out_dim=width)
        # self.cross_clip_sam = cross_clip_sam(kwargs["dual_encoder"])

        #gamma,学习的参数，用于调整模型的输出
        out_cls = configer.get("data","num_classes")

        if self.prompt == 'clip':  # clip with sam
            interaction_dim = clip_dim
        elif self.prompt == "sam":
            interaction_dim = sam_dim
        elif self.prompt == 'clipsam':
            interaction_dim = sam_dim
        else:
            raise ValueError

        self.param_multi_level = nn.Parameter(torch.ones(4))
        self.param_multi_score = nn.Parameter(torch.ones(4))

        self.embed_img_t  = Mlp(in_features=width + self.num_classes, hidden_features = 4 * (width + self.num_classes) ,out_features=interaction_dim, drop = proj_drop)


        self.fusion_score = Mlp(in_features = 2 * self.num_classes, hidden_features = 4 * (2 * self.num_classes) ,out_features=out_cls, drop = proj_drop)
        self.mlp_feats_score = Mlp(in_features= width + self.num_classes, hidden_features=4 * ( width + self.num_classes),
                                out_features=width, drop=proj_drop)

        self.interacts =  Cross_Modal_Attention(interaction_dim, width, num_heads=n_cross_head, depth=1,
                                          window_size=1,
                                          mlp_ratio=4., qkv_bias=False,
                                          qk_scale=None, drop=proj_drop,
                                          attn_drop=attn_drop, feats_fusion=feats_fusion,
                                          feats_exchange=feats_exchange)
        # self.interacts = nn.ModuleList(
        #     [
        #             Cross_Modal_Attention(interaction_dim, width, num_heads=n_cross_head, depth=1,
        #                                   window_size=1,
        #                                   mlp_ratio=4., qkv_bias=False,
        #                                   qk_scale=None, drop=proj_drop,
        #                                   attn_drop=attn_drop, feats_fusion=feats_fusion,
        #                                   feats_exchange=feats_exchange)
        #
        #             for i in range(4)
        #     ]
        # )



        # Register buffers to store positive and negative features
        self.register_buffer("mem_queue", torch.randn((out_cls, n_pts, width)))
        self.segment_queue = nn.functional.normalize(self.mem_queue, p=2, dim=2)
        self.register_buffer("mem_queue_ptr", torch.zeros(out_cls, dtype=torch.long))

        neck_dict = configer.get("network", "neck")["fusion"]
        neck_dict["params"]["num_class"] = out_cls
        self.neck = BuildHead(configer).build_head(name=neck_dict["name"], **neck_dict["params"])

        self.embed_feats = nn.Sequential(
            nn.Conv2d(width * 4, width, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(width , out_cls, kernel_size=1, stride=1, padding=0)
        )

        self.decoder = MultiMaskDecoder(
            transformer_dim = width ,
            text_dim = width,
            transformer  = TwoWayTransformer(
                depth=12,
                embedding_dim=width,
                num_heads=8,
                mlp_dim=1024
            ),
            num_multimask_outputs = out_cls,
            activation=nn.ReLU,
            n_feats = 1
        )

    def build_pyramid_features(self, features):
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        b , hw, c = features[0].shape
        for i,feature in enumerate(features):
            feature = feature.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )
            features[i] = ops[i](feature)
        return features

    def compute_v_t_score(self, t_embedding, sam_visual_embeddings, x , prompt_sam = None):

        #B, _, H, W = sam_visual_embeddings.shape
        x_orig = list(x[0:4])
        global_feat, visual_context =   x[4]
        B, C, H, W = visual_context.shape
        # sam_visual_embeddings = self.embed_sam(self.ln_sam(sam_visual_embeddings.reshape(B, -1, H*W ).permute(0,2,1))).permute(0,2,1)
        #
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_context.reshape(B, C, H * W)],dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = t_embedding.expand(B, -1, -1).float()
        # input text_embeddings: B X K X C     visual_context: B X N X C
        visual_embeddings = self.context_encoder(visual_context, text_embeddings)
        # (B, K, C)
        #text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = visual_embeddings @ text_embeddings.transpose(-2,-1)  # torch.einsum('bcn,bkc->bkn', visual_embeddings, text)  # z @ t
        #C = sam_visual_embeddings.shape[1]
        #print(prompt_sam)
        visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)
        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        score_map  = torch.einsum('bnk,bnc->bnc', visual_embeddings, score_map)  # z @ t
        if prompt_sam is not False:
            return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        else:
            return text_embeddings, x, score_map

    def  fusion_clip_to_sam(self, t_embedding, clip_feats, sam_feats, fusion = "sam_clip"):


        if fusion == 'clip':
            return clip_feats
        elif fusion == 'sam':
            return sam_feats
        sam_feats = self.norm_sam(sam_feats)
        clip_feats = self.norm_clip(clip_feats)
        # (B, K, C)

        text_embeddings = t_embedding.expand(clip_feats.shape[0], -1, -1).float()
        # input text_embeddings: B X K X C     visual_context: B X N X C
        text_embeddings = self.context_visual_attn(text_embeddings, clip_feats)  # attention(q,[z_hat,z])


        visual_embeddings = clip_feats
        text = F.normalize(text_embeddings, dim=2, p=2)
        visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_feats)
        # C = sam_visual_embeddings.shape[1]
        # # print(prompt_sam)
        # if prompt_sam is not None:
        #     if len(sam_feats.shape) == 4:
        #         B, C, H, W = sam_feats.shape
        #         sam_feats = sam_feats.reshape(B, C, H * W).permute(0, 2, 1)
        #     if prompt_sam:  # sam's features set as prompt
        #         visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_feats)
        #     else:  ## clip's features set as prompt
        #         visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_feats)[:, 1:, :]
        #         # visual_embeddings = self.proj_sam(visual_embeddings)
        #         #score_map = score_map[:, 1:, :]

        score_map = visual_embeddings @ text.transpose(-2,-1)  # torch.einsum('bcn,bkc->bkn', visual_embeddings, text)  # z @ t
        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        # if prompt_sam is not False:
        #     # 修改shape
        #     return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        # else:
        return  visual_embeddings, score_map
    def  compute_t_v_score(self, t_embedding, sam_visual_embeddings, x , prompt_sam = None):

        #B, _, H, W = sam_visual_embeddings.shape
        #x_orig = list(x[0:4])
        global_feat, visual_context = x[4]
        B, C, H, W = visual_context.shape
        # sam_visual_embeddings = self.embed_sam(self.ln_sam(sam_visual_embeddings.reshape(B, -1, H*W ).permute(0,2,1))).permute(0,2,1)
        #
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_context.reshape(B, C, H * W)],dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = t_embedding.expand(B, -1, -1).float()
        # input text_embeddings: B X K X C     visual_context: B X N X C
        text_diff = self.context_encoder(text_embeddings, visual_context)  # attention(q,[z_hat,z])
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = F.normalize(visual_context, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = visual_embeddings @ text.transpose(-2,-1)  # torch.einsum('bcn,bkc->bkn', visual_embeddings, text)  # z @ t
        #C = sam_visual_embeddings.shape[1]
        #print(prompt_sam)
        if prompt_sam is not None:
            if len(sam_visual_embeddings.shape) == 4:
                B, C, H, W = sam_visual_embeddings.shape
                sam_visual_embeddings = sam_visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1)
            if prompt_sam:# sam's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)
            else: ## clip's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)[:,1:, :]
                #visual_embeddings = self.proj_sam(visual_embeddings)
                score_map = score_map[:,1:,:]


        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        if prompt_sam is not False:
            #修改shape
            return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        else:
            return text_embeddings, visual_embeddings, score_map
    def compute_text_visual_score(self, t_embedding, x):
        '''
            text_embeddings: B X K X C
        '''
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':  # use sam to update it
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C



        # (B, K, C)
        text_embeddings = t_embedding.expand(B, -1, -1)
        #input text_embeddings: B X K X C     visual_context: B X N X C
        text_diff = self.context_decoder(text_embeddings, visual_context) # attention(q,[z_hat,z])
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text) #z @ t
        x_orig[-2] = self.embed_img_t(torch.cat([x_orig[-2], score_map], dim=1)).view(x_orig[-2].shape)

        return text_embeddings, x_orig, score_map

    def cmp_prompts(self,target):
        if target is None:
            return None

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def SCFE(self, x, t_embedding):
        # x1_embedding：编码；x1_feats，特征；x1_global：全局特征


        with torch.no_grad():
            _, sam_feats = self.image_encoder(x)
            clip_feats = self.clip_image_encoder(x)
            #visual_feats, global_feats = clip_feats[:4], clip_feats[4:]


        if self.prompt == 'clip':
            return  clip_feats
        elif self.prompt == 'sam':
            return sam_feats
        elif self.prompt=='clipsam':
            x_fuse = sam_feats
            x_os = []
            score_maps = []
            for i, (sam_feat, clip_feat) in enumerate(zip(sam_feats, clip_feats)):
                sam_feat = einops.rearrange(sam_feat, "b c h w -> b (h w) c")
                clip_feat = einops.rearrange(clip_feat, "b c h w -> b (h w) c")
                x_o, score= self.fusion_clip_to_sam(t_embedding, clip_feat, sam_feat, self.prompt)
                x_o = einops.rearrange(x_o, "b (h w) c -> b c h w", h = int(math.sqrt(x_o.shape[1])))
                x_os.append(x_fuse[i] + x_o)
                score_maps.append(score)

            #不同层次的x_os 自适应加权
            # 归一化权重
            weights_feats = F.softmax(self.param_multi_level, dim=0).view(-1, 1, 1, 1, 1)  # (L, 1, 1, 1, 1)
            weights_scores = F.softmax(self.param_multi_score, dim=0).view(-1, 1, 1, 1)  # (L, 1, 1, 1)
            # 加权融合
            x_fuse = sum(w * x for w, x in zip(weights_feats, x_os))
            score_map = sum(w * x for w, x in zip(weights_scores, score_maps))
            return x_fuse, score_map
        else:
            raise ValueError(f'Unsupported prompt {self.prompt}')
        # x_fuse = []
        # for sam_f, clip_f in zip(sam_feats, clip_feats):
        #     sam_f = einops.rearrange(sam_f, "b c h w -> b (h w) c")
        #     clip_f = einops.rearrange(clip_f, "b c h w -> b (h w) c")
        #     x_o= self.fusion_clip_to_sam(t_embedding, clip_f,sam_f, self.prompt)
        #     x_fuse.append(x_o)



    #输入x1,x2为图像
    def forward(self, x1, x2, targets = None,batched_input = None, multimask_output = False, with_amp = False ):
        #encoder text
        prompts = self.prompt_learner()
        #prompts_t2 = self.prompt_learner(timesteps = 1)# K x C x dim
        #获取已经标记化的提示,文本形式
        tokenized_prompts = self.prompt_learner.tokenized_prompts #K x C

        #无需计算梯度，避免不必要的开销
        #encoder x1 and x2
        with torch.no_grad():
            _, t_embedding = self.text_encoder(prompts, tokenized_prompts)  # K X C
            #_, t2_embedding = self.text_encoder(prompts_t2, tokenized_prompts)  # K X C
        t_embedding = self.embed_t0(t_embedding.float())
        #t2_embedding = self.embed_t1(t2_embedding.float())
        x1_orig, score_x1 = self.SCFE(x1, t_embedding)
        x2_orig, score_x2 = self.SCFE(x2, t_embedding)
        x_fuses = []
        f1 = einops.rearrange(x1_orig, "b c h w -> b (h w) c")
        f2 = einops.rearrange(x2_orig, "b c h w -> b (h w) c")
        x_fuse = self.interacts(f1, f2)  # fusion method

        # for i, (f1, f2) in enumerate(zip(x1_orig, x2_orig)):
        #     f1 = einops.rearrange(f1, "b c h w -> b (h w) c")
        #     f2 = einops.rearrange(f2, "b c h w -> b (h w) c")
        #     x_fuses.append(self.interacts[i](f1, f2)) #fusion method


        score_fusion = self.fusion_score(torch.cat((score_x1, score_x2), dim = -1))
        x_fuses = self.mlp_feats_score(torch.cat((x_fuse, score_fusion), dim = -1))

        x_fusion = self.neck(x_fuses)

        dense_embed = self.embed_feats(torch.cat((x_fusion),dim = 1))

        corse_pred = self.aux_head(dense_embed)
        pred = self.decoder([dense_embed], corse_pred)

        return {"pred": pred, "dense_embed": dense_embed, "pred_aux": corse_pred}

        # attn =torch.einsum('bkn,bcm->bmn', t1_embedding, t2_embedding)
        # attn = 1.0 - torch.softmax(attn, dim=-1)
        # t1_embedding = t1_embedding @ attn
        # t2_embedding = t2_embedding @ attn
        #
        # t_embedding = t1_embedding + t2_embedding
        #
        # pred = self.decoder([dense_embed], t_embedding)

        return  {"pred":pred, "dense_embed":dense_embed  }




if __name__ == "__main__":
    pass