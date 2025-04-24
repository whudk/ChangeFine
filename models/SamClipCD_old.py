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
# from models.decoder.BuildHead import BuildHead
# from lib.models.nets.attention import Mlp
_tokenizer = _Tokenizer()
from models.clip.model import VisionTransformer

from models.attention import Cross_Modal_Attention

from Inference.hdrpylc.hdrpylc import checkLib
from models.decoder.transformer_decoder import MaskDecoder,TwoWayTransformer
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






class ClipCD(nn.Module):
    def __init__(self,
                 configer,
                 clip_model,
                 context_length,
                 class_names,
                 visual_dim = 512,
                 visual_width=768,
                 text_dim=512,

                 feats_exchange="CCMAT",
                 feats_fusion="TBAM",  # "CONCAT", "DIFF"
                 encoder_dim=512,
                 n_cross_head=16,
                 n_cross_layer=1,
                 attn_drop=0.,
                 proj_drop=0.,
                 mask_threshold = 0.0,
                 **kwargs):
        super().__init__()

        self.dtype = clip_model.dtype

        self.text_encoder = TextEncoder(clip_model)
        for p in self.parameters():
            p.requires_grad = False

        self.prompt_learner = PromptLearner(configer, class_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = CLIPVisionTransformer(pretrained=clip_model, patch_size=16, input_resolution=512,
                                                   get_embeddings=True)
        self.context_feature = 'attention'
        self.context_length = context_length
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)



        context_decoder_params = configer.get("network", "context_decoder")["params"]
        self.context_decoder = ContextDecoder(**context_decoder_params)

        self.images_interacts = Cross_Modal_Attention(visual_dim, num_heads=n_cross_head, depth=n_cross_layer,
                                                     window_size=1,
                                                     mlp_ratio=4., qkv_bias=False,
                                                     qk_scale=None, drop=proj_drop,
                                                     attn_drop=attn_drop, feats_fusion=feats_fusion,
                                                     feats_exchange=feats_exchange)
        self.embed_img_t = Mlp(visual_width + self.num_classes, visual_width * 4, visual_width)
        self.embed_img = Mlp(visual_width * 4, visual_width * 4, encoder_dim)
        self.norm_img = nn.LayerNorm(encoder_dim)

        chg_head_dict = configer.get("network", "decoder")["head"]
        self.chg_head = BuildHead(configer).build_head(name=chg_head_dict["name"], **chg_head_dict["params"])

        self.aux_head = nn.Sigmoid()

        #denseclip type
        # context_decoder_params = configer.get("network", "text_context_encoder")
        # self.text_encoder = CLIPTextContextEncoder(**context_decoder_params)
        # context_length = self.text_encoder.context_length - self.context_length
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        # nn.init.trunc_normal_(self.contexts)
        # #
        # self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        #
        #
        #
        # context_decoder_params = configer.get("network", "context_decoder")["params"]
        # self.context_decoder = ContextDecoder(**context_decoder_params)
        # self.embed_img_t = nn.Linear(visual_dim + self.num_classes, visual_dim)
        #
        #
        # self.aux_head = nn.Linear(self.num_classes *2, 2)
        #
        # self.interacts =Cross_Modal_Attention(visual_dim, num_heads=n_cross_head, depth=n_cross_layer,
        #                              window_size=1,
        #                              mlp_ratio=4., qkv_bias=False,
        #                              qk_scale=None, drop=proj_drop,
        #                              attn_drop=attn_drop, feats_fusion=feats_fusion,
        #                              feats_exchange=feats_exchange)
        # self.embed_img_t = Mlp(visual_width + self.num_classes, visual_width * 4, visual_width)
        # self.embed_img = Mlp(visual_width * 4, visual_width * 4, encoder_dim)
        # self.norm_img = nn.LayerNorm(encoder_dim)
        #
        # chg_head_dict = configer.get("network", "decoder")["head"]
        #self.chg_head = BuildHead(configer).build_head(name=chg_head_dict["name"], **chg_head_dict["params"])
        #
        #
        # self.aux_head = nn.Sigmoid()
        self.aux_head = nn.Sequential(
            nn.LayerNorm(len(class_names) * 2),
            nn.Linear(self.num_classes * 2, 2),
            nn.GELU()
        )



    def compute_t_v_score(self, t_embedding, sam_visual_embeddings, clip_visual_embeddings):

        B, _, H, W = sam_visual_embeddings.shape
        global_feat, _ =   clip_visual_embeddings[4]

        sam_visual_embeddings = self.embed_sam(self.ln_sam(sam_visual_embeddings.reshape(B, -1, H*W ).permute(0,2,1))).permute(0,2,1)

        clip_visual_global = self.embed_clip(self.ln_clip(global_feat))

        C = sam_visual_embeddings.shape[1]

        if self.context_feature == 'attention':  # use sam to update it
            visual_context = torch.cat([clip_visual_global.reshape(B, C, 1), sam_visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = t_embedding.expand(B, -1, -1).float()
        # input text_embeddings: B X K X C     visual_context: B X N X C
        text_diff = self.context_decoder(text_embeddings, visual_context)  # attention(q,[z_hat,z])
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = F.normalize(sam_visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bcn,bkc->bkn', visual_embeddings, text)  # z @ t



        x = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=1).permute(0,2,1))

        return text_embeddings, x, score_map
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
    def forward(self, x1, x2 ):
        #encoder text
        #






        prompts = self.prompt_learner() # K x C x dim
        tokenized_prompts = self.tokenized_prompts #K x C
        _, t_embedding = self.text_encoder(prompts, tokenized_prompts) # K X C

        #encoder x1 and x2
        #x1_global = self.clip_image_encoder(x1)
        x1_orig = self.image_encoder(x1)

        x2_orig = self.image_encoder(x2)

        if self.training is False:
            t_embedding = t_embedding.float()
        t1_embedding, x1, s1 = self.compute_text_visual_score(t_embedding, x1_orig)
        t2_embedding, x2, s2 = self.compute_text_visual_score(t_embedding, x2_orig)

        _, _, h, w = x1[0].size()

        feats_left = torch.cat(x1, dim=1)

        feats_right = torch.cat(x2, dim=1)

        feats_left = self.norm_img(self.embed_img(feats_left))
        feats_right = self.norm_img(self.embed_img(feats_right))
        fusions = []
        fusions.append(self.images_interacts(feats_left, feats_right))

        pred, _ = self.chg_head(fusions)


        b,c,h,w = s1.shape
        corse_pred = torch.cat((s1,s2),dim=1).reshape(b,2 * c,h*w).permute(0,2,1)
        pred_aux = self.aux_head(corse_pred).permute(0,2,1).reshape(b,-1,h,w)
        #pred_aux = nn.functional.pairwise_distance(s1, s2, p=2, keepdim=True)
        # pred_aux = self.aux_head(s_diff)

        return pred, pred_aux


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



class SamClipCD(nn.Module):
    def __init__(self,
                 configer,
                 clip_model,
                 sam_model,
                 context_length,
                 class_names,
                 input_imgsz = 512,
                 sam_dim = 256,
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

        self.dtype = clip_model.dtype

        self.mask_threshold = mask_threshold
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = sam_model.image_encoder
        for param in self.parameters():
            param.requires_grad = False

        self.prompt = prompt
        self.prompt_sam = False
        if self.prompt in ["clip","sam"] :
            if self.prompt == "sam":
                self.prompt_sam = True
        else:
             self.prompt_sam = None
        #self.prompt_encoder = PromptEncoder(visual_dim,[32,32],[512,512],16)
        #self.mask_decoder =MaskDecoder(num_multimask_outputs=2,transformer=TwoWayTransformer(depth=2,embedding_dim=visual_dim,mlp_dim=2048,num_heads=8,),transformer_dim=visual_dim,iou_head_depth=3,iou_head_hidden_dim=256,)
        # self.ln_clip = nn.LayerNorm(clip_dim)
        # self.ln_sam = nn.LayerNorm(sam_dim)
        # self.embed_clip = nn.Linear(clip_dim, visual_dim)
        # self.embed_sam = nn.Linear(sam_dim, visual_dim)
        self.clip_image_encoder = CLIPVisionTransformer(input_resolution=256 ,pretrained=clip_model, patch_size= 16, get_embeddings=True)


        self.prompt_learner = PromptLearner(configer, class_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.context_feature = 'attention'
        self.context_length = context_length
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)



        out_dim = clip_dim if self.prompt == "sam" else sam_dim

        self.cross_clip_sam = cross_clip_sam(
                                            transformer_width=512,
                                             transformer_heads=8,
                                             transformer_layers=6,
                                             visual_dim=sam_dim,
                                             out_dim=out_dim)

        self.proj_sam = nn.Sequential(
            nn.Linear(sam_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.ReLU()
        )

        #denseclip type
        # context_decoder_params = configer.get("network", "text_context_encoder")
        # self.text_encoder = CLIPTextContextEncoder(**context_decoder_params)
        # context_length = self.text_encoder.context_length - self.context_length
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        # nn.init.trunc_normal_(self.contexts)
        #
        out_cls = configer.get("data","num_classes")
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        self.vv_gamma = nn.Parameter(torch.ones(out_dim))

        context_decoder_params = configer.get("network", "context_decoder")["params"]
        self.context_encoder = cross_clip_sam(
                                             clip_dim = clip_dim,
                                             transformer_width=512,
                                             transformer_heads=8,
                                             transformer_layers=1,
                                             visual_dim= clip_dim,
                                             out_dim= clip_dim)
        # self.embed_img_t =nn.Sequential(
        #     nn.LayerNorm(visual_dim + self.num_classes),
        #     nn.Linear(visual_dim + self.num_classes, width),
        #     nn.GELU(),
        # )

        self.embed_img_t  = Mlp(in_features=visual_dim + self.num_classes, hidden_features = 4 * (visual_dim + self.num_classes) ,out_features=width, drop = proj_drop)

        self.embed_feats = nn.Sequential(
            nn.Conv2d(visual_dim * 4, visual_dim, kernel_size=3, stride=1,padding = 1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU()
        )


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
        self.register_buffer("mem_queue", torch.randn((out_cls, n_pts, 512)))
        self.segment_queue = nn.functional.normalize(self.mem_queue, p=2, dim=2)
        self.register_buffer("mem_queue_ptr", torch.zeros(out_cls, dtype=torch.long))
        # self.embed_img_t = Mlp(visual_width + self.num_classes, visual_width * 4, visual_width)
        # self.embed_img = Mlp(visual_width * 4, visual_width * 4, encoder_dim)
        # self.norm_img = nn.LayerNorm(encoder_dim)
        #
        # chg_head_dict = configer.get("network", "decoder")["head"]
        #self.chg_head = BuildHead(configer).build_head(name=chg_head_dict["name"], **chg_head_dict["params"])
        #
        #
        # self.aux_head = nn.Sigmoid()
        # self.aux_head = nn.Sequential(
        #     nn.LayerNorm(len(class_names)),
        #     nn.Linear(len(class_names), 2),
        #     nn.GELU()
        # )
        neck_dict = configer.get("network", "neck")["fusion"]
        neck_dict["params"]["num_class"] = out_cls
        self.neck = BuildHead(configer).build_head(name=neck_dict["name"], **neck_dict["params"])


        self.decoder = MaskDecoder(
            transformer_dim = visual_dim,
            transformer  = TwoWayTransformer(
                depth=12,
                embedding_dim=visual_dim,
                num_heads=8,
                mlp_dim=visual_dim * 4
            ),
            num_multimask_outputs = out_cls,
            activation=nn.ReLU
        )



        # self.aux_head = nn.Sequential(
        #     nn.LayerNorm(len(class_names) * 2),
        #     nn.Linear(self.num_classes * 2, out_cls),
        #     nn.GELU()
        # )
        #self.conv_last = nn.Conv2d(visual_dim, 2, kernel_size=1)

        # if 'train' in configer.get('phase'):
        #     checkLib(23)
        # else:
        #     checkLib(22)

    #self.register_buffer("positive_features", torch.zeros((n_pts, width)))
    #self.register_buffer("negative_features", torch.zeros((n_pts, width)))



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

        if prompt_sam is not False:
            return text_embeddings, visual_embeddings[:, 1:, :], score_map[:, 1:, :]
        else:
            return text_embeddings, x, score_map


    def compute_t_v_score(self, t_embedding, sam_visual_embeddings, x , prompt_sam = None):

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
            B, C, H, W = sam_visual_embeddings.shape
            sam_visual_embeddings = sam_visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1)
            if prompt_sam:# sam's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)
            else: ## clip's features set as prompt
                visual_embeddings = self.cross_clip_sam(visual_embeddings, sam_visual_embeddings)[:,1:, :]
                visual_embeddings = self.proj_sam(visual_embeddings)
                score_map = score_map[:,1:,:]





        visual_embeddings = self.embed_img_t(torch.cat([visual_embeddings, score_map], dim=-1))

        if prompt_sam is not False:
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
    def forward(self, x1, x2, targets = None,batched_input = None, multimask_output = False, with_amp = False ):
        #encoder text
        #




        prompts = self.prompt_learner() # K x C x dim
        tokenized_prompts = self.tokenized_prompts #K x C


        #encoder x1 and x2
        #x1_global = self.clip_image_encoder(x1)
        with torch.no_grad():
            _, t_embedding = self.text_encoder(prompts, tokenized_prompts)  # K X C
            x1_embedding, x1_feats = self.image_encoder(x1)
            x1_global = self.clip_image_encoder(x1)


            x2_embedding, x2_feats = self.image_encoder(x2)
            x2_global = self.clip_image_encoder(x2)

        #debug



        if self.training is False:
            t_embedding  = t_embedding.float()
        t1_embedding, x1, s1 = self.compute_v_t_score(t_embedding, x1_embedding, x1_global, self.prompt_sam)
        t2_embedding, x2, s2 = self.compute_v_t_score(t_embedding, x2_embedding, x2_global, self.prompt_sam)

        # if self.text_head:
        #     x = [text_embeddings, ] + x_orig
        # else:
        #     x = x_orig


        if self.prompt_sam:
            b, hw,c = x1.size()
            x1_orig = x1_feats
            x2_orig = x2_feats



            x1_orig[3] = x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )#x1_orig[3] + x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )
            x2_orig[3] = x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )#x2_orig[3] + x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )

           # _, _, h, w = x1[0].size()
            B, hw, c = x2.size()
            #batched_input = torch.stack( [for x in batched_input], dim=0)

        else: # clip_prompt
            b, hw, c = x1.size()
            # x1_orig = x1_feats
            # x2_orig = x2_feats
            x1_orig = list(x1_global[0:4])
            x2_orig = list(x2_global[0:4])
            x1_orig[3] = x1_orig[3] +  x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
            x2_orig[3] = x2_orig[3] + x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

            # x1_orig[3] = x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))  # x1_orig[3] + x1.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )
            # x2_orig[3] = x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))  # x2_orig[3] + x2.reshape(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)) )
        x_fuses = []
        for i in range(4):
            b, c, h, w = x1_orig[i].size()
            f1 = einops.rearrange(x1_orig[i], "b c h w -> b (h w) c")
            f2 = einops.rearrange(x2_orig[i], "b c h w -> b (h w) c")
            # if i == 3:
            #     f1 = f1  + self.context_image(f1, t1_embedding)
            #     f2 = f2  + self.context_image(f2, t2_embedding)
            x_fuses.append(self.interacts[i](f1, f2))





        #embed_feats = self.embed_feats(torch.cat((x_fuses), dim = -1))

        outputs = []
        dense_embed = self.neck(x_fuses)
        dense_embed = self.embed_feats(dense_embed)

        diff_t_embedding = t2_embedding + (t2_embedding - t1_embedding)

        pred = self.decoder(dense_embed, diff_t_embedding)
        # pred = self.conv_last(dense_embed)
        # corse_pred = torch.cat((s1, s2), dim=-1)
        # pred_aux = self.aux_head(corse_pred).permute(0, 2, 1)
        # N = pred_aux.shape[-1]
        # H = W = int(math.sqrt(N))
        # #embed_feats = einops.rearrange(embed_feats, "b (h w)  c -> b c h w", h=H, w=W)
        # pred_aux = einops.rearrange(pred_aux,"b c (h w) -> b c h w",  h=H, w=W )
        # pred_aux = nn.functional.pairwise_distance(s1, s2, p=2, keepdim=True)
        # pred_aux = self.aux_head(s_diff)
        # outputs.append(
        #     {
        #         "pred":pred,
        #         "pred_aux": pred_aux,
        #     }
        # )
        #update positive_features and negtive_features num 10 according dense_embed and target
        # if self.training:
        #     self.update_features_based_on_target(targets,dense_embed)

        return  {"pred":pred, "dense_embed":dense_embed  }

        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        #     if "point_coords" in image_record:
        #         points = (image_record["point_coords"], image_record["point_labels"])
        #     else:
        #         points = None
        #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #         points=points,
        #         boxes=image_record.get("boxes", None).to(image_embeddings.device) if "boxes" in image_record else None,
        #         masks=image_record.get("mask_inputs", None).to(image_embeddings.device) if "mask_inputs" in image_record else None,
        #     )
        #     low_res_masks, iou_predictions = self.mask_decoder(
        #         image_embeddings=curr_embedding.unsqueeze(0),
        #         image_pe=self.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=multimask_output,
        #     )
        #     masks = self.postprocess_masks(
        #         low_res_masks,
        #         input_size=[512,512],
        #         original_size=512,
        #     )
        #     #masks = torch.masks > self.mask_threshold
        #     outputs.append(
        #         {
        #             "pred":masks,
        #             "masks": masks,
        #             "iou_predictions": iou_predictions,
        #             "low_res_logits": low_res_masks,
        #         }
        #     )


        return outputs






if __name__ == "__main__":
    from utils.tools.logger import Logger
    from utils.tools.configer import Configer

    from models.sam.build_sam import build_sam_vit_b
    from dataset.loader.clipsamDataset import SamClipCD_dataset
    from torch.utils.data import DataLoader
    from models import clip
    from dataset.tools.cv2_aug_transform_chg import CV2AugCompose_CHG
    from dataset.data_loader import DataLoader


    config = "C:\dengkai\guangzhou0415\guangzhou0415\clip_sam_cd\config\clip_change.json"
    classnames = ["background","Farm","Forest","Garden","Grassland","Buildings","Stuctures","Roads","Filled","Water"]


    config = Configer(configs=config)

    #build dataset
    json_file = r"C:\dengkai\data\1m2m\1m2m_train.json"
    transforms = CV2AugCompose_CHG(config, split="train")
    data_loader = DataLoader(config)

    data_loader = data_loader.build_loader(json_file)

    #load sam
    sam_b = build_sam_vit_b(checkpoint=r"F:\sam_checkpoints\sam_vit_b.pth").eval()
    #load clip
    clip_path = "ViT-B/16"
    clip_model, _ = clip.load(clip_path, "cuda")




    #init sam_clip_cd
    sam_clip_cd = SamClipCD(config,clip_model,sam_model = sam_b,context_length=77, class_names= classnames).cuda()



    for oldimg, newimg, target, img_records  in data_loader:


        res = sam_clip_cd(oldimg.float().cuda(), newimg.float().cuda(), batched_input = img_records)

        pass







    num_parameters = sum(p.numel() for p in sam_clip_cd.parameters() if p.requires_grad)
    print(f'#Params: {num_parameters}')
    pass
    # model = Cross_ChannelsAttention(dim=768,num_heads=6)
    # out =model(x1,x2)


    #print(out.shape)