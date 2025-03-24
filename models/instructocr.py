# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .transformer_decoder import build_transformer
from collections import OrderedDict
from .position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        x, att_maps = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )

        return x, att_maps


class ResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, width=64, d_model=256):
        super().__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    
    def init_parameters(self):

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3) 
        x5, att_maps = None, None
        return [x1, x2, x3, x4], x5, att_maps

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, mask):
        mask = mask.to(device=x.device) if mask is not None else None
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask, attn_mask=self.attn_mask)[0]
        
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: list):
        x, mask = x
        x = x + self.attention(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, mask]


class TextTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)

def _build_vision_encode(embed_dim, backbone, vision_width, vision_layers, d_model
):
    act_layer = nn.GELU
    if backbone == 'ResNet':
        vision_heads = vision_width * 32 // 64

        visual = ResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                width=vision_width,
                d_model=d_model
        )

    else:
        raise "No current encode..."
    
    return visual

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BERTTokenizer():
    def __init__(self, bert_path, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.max_length = max_length

    def __call__(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens
    
class BERTEmbedder():
    def __init__(self, bert_path):
        super().__init__()
        from transformers import BertModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = BertModel.from_pretrained(bert_path).to(self.device)

    def forward(self, tokens):
        outputs = self.transformer(input_ids=tokens)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
    
    def __call__(self, tokens):
        return self.forward(tokens)


class InstructOCR(nn.Module):
    def __init__(self, args, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes

        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_channel = 2048

        self.num_classes = num_classes
        self.embed_dim = 512
        self.backbone = "ResNet"
        self.vision_width = 64
        self.vision_layers = [3,4,6,3]
        self.d_model = 256
        self.vis_d_model = 1024
        self.visual = _build_vision_encode(self.embed_dim, self.backbone, self.vision_width, self.vision_layers, self.vis_d_model)
        self.position_embedding = self.build_position_embedding('sine', hidden_dim)

        self.context_length = 77
        self.transformer_width = 512
        self.transformer_layers = 6
        self.transformer_heads = 8
        self.tokenizer = BERTTokenizer(args.bert_path)
        self.bert_embedder = BERTEmbedder(args.bert_path)

        self.text_projection = nn.Parameter(torch.empty(768, 256))
        self.final = nn.Conv2d(2048, 512, kernel_size=1, stride = 1)
        self.pool = nn.AdaptiveAvgPool1d(self.context_length)
        self.initialize_parameters()

    def dtype(self):
        return self.visual.conv1.weight.dtype

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.final.weight, nonlinearity='relu')
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, 0)       
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
    
    def load_pretrained_weights(self):
        checkpoint = torch.jit.load('./RN50.pt', map_location='cpu').float().state_dict()
        text_transformer_state_dict = {}
        visual_state_dict = {}

        for k in checkpoint.keys():
            if k.startswith('transformer.'):
                text_transformer_state_dict[k.replace('transformer.', '')] = checkpoint[k]
            elif k.startswith('visual.'):
                visual_state_dict[k.replace('visual.', '')] = checkpoint[k]
        if 'attnpool.positional_embedding' in visual_state_dict:
            del visual_state_dict['attnpool.positional_embedding']

        self.visual.load_state_dict(visual_state_dict)

    def build_position_embedding(self, type, hidden_dim):
        N_steps = hidden_dim // 2
        if type in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif type in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {type}")
        return position_embedding
    
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    def encode_text(self, text):

        tokens = self.tokenizer(text)
        x = self.bert_embedder(tokens)
        x = x @ self.text_projection

        return x   

    #def forward(self, samples: NestedTensor, sequence, sequence_reg, text, text_length = 25):
    def forward(self, samples: NestedTensor, sequence_reg, text, text_length = 25):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - sequence: the input sequence for locating in the first decoder layer.
               - sequence_reg: the input sequence for recognizing in the second decoder layer.

            It returns a dict with the following elements:
               - out: prediction of location and recognition 
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        images = samples.tensors
        all_feature, image_ori, att_maps = self.visual(images)
        encoded_texts = self.encode_text(text)

        decode_image_feature = self.final(all_feature[-1])
        image_masks = samples.mask
        mask = F.interpolate(image_masks[None].float(), size=decode_image_feature.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(decode_image_feature, mask)

        if self.training:
            seq = self.transformer(decode_image_feature, mask, pos, sequence_reg, encoded_texts)
            return seq
        else:
            seq, probs = self.transformer(decode_image_feature, mask, pos, sequence_reg, encoded_texts)
            return seq, probs

def build(args):

    device = torch.device(args.device)

    transformer = build_transformer(args)

    num_classes = args.num_classes
    model = InstructOCR(args, transformer, num_classes)

    weight = torch.ones(num_classes)
    weight[args.end_index] = 0.01; weight[args.noise_index] = 0.01
    criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
    criterion.to(device)

    return model, criterion
