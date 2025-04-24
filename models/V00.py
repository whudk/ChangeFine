import math
from typing import Any, Dict, List, Tuple

import einops
import torch.nn as nn
import torch
from models.clip.clip import tokenize
from models.clip import clip
from models.clip.model import CLIPVisionTransformer, ContextDecoder, CLIPTextContextEncoder,cross_clip_sam
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

    def forward(self):
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

        return prompts

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, attn_drop = 0.,dropout=0.):
        super().__init__()
        inner_dim = query_dim
        context_dim = default(context_dim, query_dim)
        head_dim = context_dim // heads
        self.scale = head_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    def forward(self, x,  context):
        B, N, C = x.shape


        h = self.heads

        q = self.to_q(x)

        k = self.to_k(context)
        v = self.to_v(context)

        q = q * self.scale  # , q2 * self.scale

        q = einops.rearrange(q, 'b  n  (h1  c) -> b  h1 n c', h1 = self.heads)
        k = einops.rearrange(k, 'b n  (h1   c) -> b  h1 n c', h1 = self.heads)
        v = einops.rearrange(v, 'b n  (h1   c) -> b  h1 n c', h1 = self.heads)

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)


        #x = attn @ v
        x = (attn @ v).transpose(1, 2)
        x = einops.rearrange(x, ' b n h1 c -> b  n (h1 c)', h1 = self.heads)
        x = self.to_out(x)
        return  x#, attn


        # self.model = SamClipCD(
        #     self.configer,
        #     clip_model=clip_model,
        #     sam_model= sam_model,
        #     context_length=77,
        #     class_names= class_names,
        #     token_embed_dim= 512,
        #     text_dim= 512,
        #     prompt= self.configer.get("network", "prompt"),
        #     sam_dim = 768,
        # )

class SamClipCD(nn.Module):
    def __init__(self,
                 configer,
                 clip_model,
                 sam_model,
                 context_length,
                 class_names,
                 sam_dim = 1024,
                 clip_dim = 512,
                 visual_dim = 512,
                 text_dim=512,
                 width = 768,
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

        #如果prompt为sam，则prompt_sam为true,否则为None
        self.prompt = prompt
        self.prompt_sam = False
        if self.prompt in ["clip","sam"] :
            if self.prompt == "sam":
                self.prompt_sam = True
        else:
             self.prompt_sam = None

        #初始化图像编码器，输入大小为256，patch_size为图像patch,即小区域提取特征
        self.clip_image_encoder = CLIPVisionTransformer(input_resolution=256 ,pretrained=clip_model, patch_size= 16, get_embeddings=True)


        #prompt_learner，学习提示词的模块，将class_names转化为适合CLIP的提示,一般为数值序列
        self.prompt_learner = PromptLearner(configer, class_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        #texts通过tokenize函数将class_names转化为文本，num_classes为类别数量
        self.context_feature = 'attention'
        self.context_length = context_length
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)


        #交叉注意力模块
        out_dim = sam_dim if self.prompt == "sam" else clip_dim
        #加载sam_model时为256，不加载时为768
        self.cross_clip_sam = cross_clip_sam(
                                             clip_dim = 512,
                                             sam_dim = 256,
                                             transformer_width=512,
                                             transformer_heads=8,
                                             transformer_layers=6,
                                             visual_dim=sam_dim,
                                             out_dim=out_dim)
        # self.cross_clip_sam = cross_clip_sam(kwargs["dual_encoder"])

        #SAM特征映射层，将SAM模型特征映射到与CLIP一个维度，
        self.proj_sam = nn.Sequential(
            nn.Linear(sam_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.ReLU()
        )

        #gamma,学习的参数，用于调整模型的输出
        out_cls = configer.get("data","num_classes")
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.vv_gamma = nn.Parameter(torch.ones(out_dim))

        context_decoder_params = configer.get("network", "context_decoder")["params"]
        self.context_encoder = cross_clip_sam(
                                             clip_dim = clip_dim,
                                             transformer_width=512,
                                             transformer_heads=8,
                                             transformer_layers=6,
                                             visual_dim= clip_dim,
                                             out_dim= clip_dim)


        self.embed_img_t  = Mlp(in_features=out_dim + self.num_classes, hidden_features = 4 * (out_dim + self.num_classes) ,out_features=width, drop = proj_drop)


        self.interacts = nn.ModuleList(
            [
                    Cross_Modal_Attention(width, visual_dim, num_heads=n_cross_head, depth=1,
                                          window_size=1,
                                          mlp_ratio=4., qkv_bias=False,
                                          qk_scale=None, drop=proj_drop,
                                          attn_drop=attn_drop, feats_fusion=feats_fusion,
                                          feats_exchange=feats_exchange)

                    for i in range(4)
            ]
        )



        # Register buffers to store positive and negative features
        self.register_buffer("mem_queue", torch.randn((out_cls, n_pts, visual_dim)))
        self.segment_queue = nn.functional.normalize(self.mem_queue, p=2, dim=2)
        self.register_buffer("mem_queue_ptr", torch.zeros(out_cls, dtype=torch.long))

        neck_dict = configer.get("network", "neck")["fusion"]
        neck_dict["params"]["num_class"] = out_cls
        self.neck = BuildHead(configer).build_head(name=neck_dict["name"], **neck_dict["params"])

        self.embed_feats = nn.Sequential(
            nn.Conv2d(visual_dim * 4, visual_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU()
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(visual_dim , out_cls, kernel_size=1, stride=1, padding=0)
        )

        self.decoder = MultiMaskDecoder(
            transformer_dim = visual_dim ,
            text_dim = visual_dim,
            transformer  = TwoWayTransformer(
                depth=12,
                embedding_dim=visual_dim,
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
        if prompt_sam is not None:
            B, C, H, W = sam_visual_embeddings.shape
            sam_visual_embeddings = sam_visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1)
            if prompt_sam:# sam's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)
            else: ## clip's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)[:,1:, :]
                visual_embeddings = self.proj_sam(visual_embeddings)
                score_map = score_map[:,1:,:]


        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        score_map  = torch.einsum('bnk,bnc->bnc', visual_embeddings, score_map)  # z @ t
        if prompt_sam is not False:
            return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        else:
            return text_embeddings, x, score_map


    def  compute_t_v_score(self, t_embedding, sam_visual_embeddings, x , prompt_sam = None):

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

    #输入x1,x2为图像
    def forward(self, x1, x2, targets = None,batched_input = None, multimask_output = False, with_amp = False ):
        #encoder text
        prompts = self.prompt_learner() # K x C x dim
        #获取已经标记化的提示,文本形式
        tokenized_prompts = self.tokenized_prompts #K x C

        #无需计算梯度，避免不必要的开销
        #encoder x1 and x2
        with torch.no_grad():
            _, t_embedding = self.text_encoder(prompts, tokenized_prompts)  # K X C
            #x1_embedding：编码；x1_feats，特征；x1_global：全局特征
            x1_embedding, x1_feats = self.image_encoder(x1)
            x1_global = self.clip_image_encoder(x1)

            x2_embedding, x2_feats = self.image_encoder(x2)
            x2_global = self.clip_image_encoder(x2)

        #debug
        #判断是否为训练模式
        if self.training is False:
            t_embedding  = t_embedding.float()

        #得到文本编码，图像特征，以及他们之间的得分
        t1_embedding, x1, s1 = self.compute_t_v_score(t_embedding, x1_embedding, x1_global, self.prompt_sam)
        t2_embedding, x2, s2 = self.compute_t_v_score(t_embedding, x2_embedding, x2_global, self.prompt_sam)

        #prompt_sam为True
        if self.prompt_sam:
            b, hw, c = x1.size()
            #对x1,x2的特征进行保存
            x1_orig = x1_feats
            x2_orig = x2_feats
            x1_orig[3] = x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
            x2_orig[3] = x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
            B, hw, c = x2.size()

        else:  # clip_prompt
            b, hw, c = x1.size()
            x1_orig = list(x1_global[0:4])
            x2_orig = list(x2_global[0:4])
            x1_orig[3] = x1_orig[3] + x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
            x2_orig[3] = x2_orig[3] + x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

        x_fuses = []
        for i in range(4):
            b, c, h, w = x1_orig[i].size()
            f1 = einops.rearrange(x1_orig[i], "b c h w -> b (h w) c")
            f2 = einops.rearrange(x2_orig[i], "b c h w -> b (h w) c")
            x_fuses.append(self.interacts[i](f1, f2)) #fusion method

        x_fusion = self.neck(x_fuses)

        dense_embed = self.embed_feats(torch.cat((x_fusion),dim = 1))

        corse_pred = self.aux_head(dense_embed)
        N = corse_pred.shape[-1]
        H = W = int(math.sqrt(N))
        pred_aux = einops.rearrange(corse_pred, "b c (h w) -> b c h w", h=H, w=W)
        # x1_list = self.neck(x1_orig)
        # x2_list = self.neck(x2_orig)
        pred = self.decoder([dense_embed], corse_pred)

        return {"pred": pred, "dense_embed": dense_embed, "pred_aux": pred_aux}

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