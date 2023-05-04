import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url) # root="/media/data2/MLIC_pretrained/CLIP"

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    # model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.MODEL.CAPTION.n_ctx
        ctx_init = cfg.MODEL.CAPTION.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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
            if cfg.MODEL.CAPTION.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial negtive context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        
        # temperature = torch.tensor(4.6, dtype=dtype)  # 
        # temperature = torch.tensor(4.24, dtype=dtype)  # 70
        temperature = torch.tensor(3.91, dtype=dtype)  # 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS



        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS
        
        # print(prompts)                       ['X X X X X X X X X X X X X X X X aeroplane.', 'X X X X X X X X X X X X X X X X bicycle.', 'X X X X X X X X X X X X X X X X bird.', 'X X X X X X X X X X X X X X X X boat.', 'X X X X X X X X X X X X X X X X bottle.', 'X X X X X X X X X X X X X X X X bus.', 'X X X X X X X X X X X X X X X X car.', 'X X X X X X X X X X X X X X X X cat.', 'X X X X X X X X X X X X X X X X chair.', 'X X X X X X X X X X X X X X X X cow.', 'X X X X X X X X X X X X X X X X diningtable.', 'X X X X X X X X X X X X X X X X dog.', 'X X X X X X X X X X X X X X X X horse.', 'X X X X X X X X X X X X X X X X motorbike.', 'X X X X X X X X X X X X X X X X person.', 'X X X X X X X X X X X X X X X X pottedplant.', 'X X X X X X X X X X X X X X X X sheep.', 'X X X X X X X X X X X X X X X X sofa.', 'X X X X X X X X X X X X X X X X train.', 'X X X X X X X X X X X X X X X X tvmonitor.']
        # print(len(prompts))                  20
        # print(tokenized_prompts.shape)       torch.Size([20, 77])
        # print(embedding.shape)               torch.Size([20, 77, 512])
        # print(self.token_prefix.shape)       torch.Size([20, 1, 512])
        # print(self.token_suffix.shape)       torch.Size([20, 60, 512])
        # print(self.token_suffix_nocls.shape) torch.Size([20, 60, 512])
                

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.MODEL.CAPTION.class_token_position

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_neg = self.ctx_neg
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_neg, # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_neg,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
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
        return prompts, prompts_neg, self.temperature, self.spatial_T


class DualCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model
        self.dtype = clip_model.dtype

        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)

        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias

        self.attnpool = self.model.visual.attnpool
        self.linear_dim = self.model.visual.attnpool.c_proj.bias.shape[0]


        self.if_pos = False
        if self.if_pos == False:
            del self.model.visual.attnpool.positional_embedding
        del self.model
        del clip_model


    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    def forward(self, image, norm=True):
        image_feat = self.encode_image(image)
        b, c, h, w = image_feat.shape
        x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)

        x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
        x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
        
        image_features = x
        image_feature_, _ = self.attnpool(image_feat, if_pos=self.if_pos)
        # ===============================================================

        prompts, prompts_neg, temperature, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_neg = self.text_encoder(prompts_neg, tokenized_prompts)
    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)            # without attention image feature
        image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)            # with attention image feature

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)               # postive text features
        text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)   # negtive text features

        logit_scale = temperature.exp() 
        
        logits = image_features @ text_features.t()                                             # postive without attention  #  HW * B * C,  cls * C,  HW * B * cls
        logits_neg = logit_scale * image_features @ text_features_neg.t()                       # negtive without attention  #  HW * B * C,  cls * C,  HW * B * cls
        
        logits_g = logit_scale * image_feature_ @ text_features.t()                             # postive with attention      # B * C,  cls * C,  B * cls
        logits_neg_g = logit_scale * image_feature_ @ text_features_neg.t()                     # negtive with attention      # B * C,  cls * C,  B * cls

        prob_ = torch.nn.functional.softmax(logits * spatial_T.exp(), dim=0)
        logits = torch.sum(logit_scale * logits * prob_, dim=0)
        logits_neg = torch.sum(logits_neg * prob_, dim=0)
        
        logits_ = logits - logits_neg         # temperature   spatial_T                         # logits without attention
        logits_g = logits_g - logits_neg_g    # temperature                                     # logits with attention

        return logits_g, logits_, image_features, text_features # self.addtmp

def build_DualCLIP(cfg):
    classnames = cfg.DATA.classnames
    clip_model = load_clip_to_cpu(cfg)
    
    if cfg.TRAIN.amp is True:
        clip_model.float()

    clip_model.float()
       

    model = DualCLIP(cfg, classnames, clip_model, return_interm_layers=False)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    return model


def do_forward_and_criterion(cfg, images, target, model, criterion):
    output_g, output, _, _ = model(images)
    loss = criterion[cfg.LOSS.loss_mode](output, target)
    # loss = criterion[cfg.LOSS.loss_mode](output, target) # + 0.0 * criterion[cfg.LOSS.loss_mode](output_g, target)

    if cfg.LOSS.loss_dev > 0:
        loss *= cfg.LOSS.loss_dev
    return output, loss