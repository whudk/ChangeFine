import torch.nn as nn
import torch
from lib.models.clip.clip import tokenize
from lib.models.clip import clip
from lib.models.clip.model import CLIPVisionTransformer, ContextDecoder, CLIPTextContextEncoder
from lib.models.nets.attention import CrossAttention_Block
from lib.models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
from lib.models.decoder.BuildHead import BuildHead
from lib.models.nets.attention import Mlp
_tokenizer = _Tokenizer()





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
                embedding = clip_model.token_embedding(prompt).type(dtype)
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
class PF_CHANGE_CLIP(nn.Module):
    def __init__(self,
                 configer,
                 clip_model,
                 context_length,
                 class_names,
                 visual_width=768,
                 text_dim=512,

                 #feats_fusion
                 feats_exchange = "CCMAT",
                 feats_fusion = "TBAM",#"CONCAT", "DIFF"
                 encoder_dim = 512,
                 n_cross_head = 16,
                 n_cross_layer = 1,
                 attn_drop = 0.,
                 proj_drop = 0.,
                 **kwargs):
        super().__init__()

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




        #denseclip type
        # context_decoder_params = configer.get("network", "text_context_encoder")
        # self.text_encoder = CLIPTextContextEncoder(**context_decoder_params)
        # context_length = self.text_encoder.context_length - self.context_length
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        # nn.init.trunc_normal_(self.contexts)
        #
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)



        context_decoder_params = configer.get("network", "context_decoder")["params"]
        self.context_decoder = ContextDecoder(**context_decoder_params)

        self.images_interacts =CrossAttention_Block(encoder_dim, num_heads=n_cross_head, depth=n_cross_layer,
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
        # self.aux_head = nn.Sequential(
        #     nn.LayerNorm(len(class_names)),
        #     nn.Linear(len(class_names), 2),
        #     nn.GELU()
        # )
        self.dtype = torch.float32

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


    def forward(self, x1, x2):
        #encoder text
        #
        prompts = self.prompt_learner() # K x C x dim
        tokenized_prompts = self.tokenized_prompts #K x C
        _, t_embedding = self.text_encoder(prompts, tokenized_prompts) # K X C




        #encoder x1 and x2
        x1_orig = self.image_encoder(x1)

        x2_orig = self.image_encoder(x2)

        if self.training is False:
            t_embedding  = t_embedding.float()
        t1_embedding, x1, s1 = self.compute_text_visual_score(t_embedding, x1_orig)
        t2_embedding, x2, s2 = self.compute_text_visual_score(t_embedding, x2_orig)

        _, _, h, w = x1[0].size()


        feats_left = torch.cat(x1, dim=1)




        feats_right = torch.cat(x2, dim=1)



        feats_left = self.norm_img(self.embed_img(feats_left))
        feats_right = self.norm_img(self.embed_img(feats_right))
        fusions = []
        fusions.append(self.images_interacts(feats_left, feats_right))



        pred,_ = self.chg_head(fusions)



        pred_aux = nn.functional.pairwise_distance(s1,s2,p = 2, keepdim=True)
        #pred_aux = self.aux_head(s_diff)

        return  pred, pred_aux





if __name__ == '__main__':
    from lib.utils.tools.configer import Configer

    config = "D:\dengkai\code\dengkai_DLNEW\configs\RSdata\SEMI_CHG_ADD.json"
    image_1 = torch.randn((1, 3, 512, 512), dtype=torch.float32).cuda()
    image_2 = torch.randn((1, 3, 512, 512), dtype=torch.float32).cuda()
    configer = Configer(configs=config)


    # model = Cross_ChannelsAttention(dim=768,num_heads=6)
    # out =model(x1,x2)

    models1 = Attention(dim=768,num_heads=6)
    out = models1(x1)
    print(out.shape)