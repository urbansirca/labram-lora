from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import wandb
import math

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=128, num_channels=45):
        super().__init__()

        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(self.num_channels, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)

        self.projection = nn.Sequential(
            nn.Conv2d(128, embed_dim, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.chan_embed = nn.Embedding(45, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, C, T)
        B, _ ,C, T = x.size()
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size,dropout=0.5):
        super().__init__(*[TransformerEncoderBlock(emb_size,drop_p=dropout) for _ in range(depth)])


class decoder(nn.Module):  
    def __init__(self, emb_size=64, depth=2, pretrain=None,**kwargs):
        super().__init__()
        self.transformer = TransformerEncoder(depth, emb_size)
    def forward(self, x):
        x = self.transformer(x)      # [batch_size, seq_length, emb_size]
        return x

class decoder_fft(nn.Module): 
    def __init__(self, emb_size=64, depth=2, pretrain=None,**kwargs):
        super().__init__()
        self.transformer = TransformerEncoder(depth, emb_size)
        self.pro= nn.Linear(emb_size, 3*2)  
    def forward(self, x):
        out = self.transformer(x)      # [batch_size, seq_length, emb_size]
        out = self.pro(torch.mean(out, dim=1))  # [batch_size, seq_length, 3*2]
        return out

class mlm_mask(nn.Module):  
    def __init__(self, emb_size=128, depth=6, n_classes=2,mask_ratio=0.5, pretrain=None,pretrainmode=False):
        super().__init__()
        self.pretrainmode = pretrainmode
        self.embedding = PatchEmbedding(embed_dim=emb_size)
        self.transformer = TransformerEncoder(depth, emb_size,dropout=0.5)
        self.clshead = nn.Linear(emb_size,n_classes)
        self.mask_ratio = mask_ratio
        if pretrain is not None:
            self.init_from_pretrained(pretrain)
        
        if self.pretrainmode:
            self.mask_token = nn.Parameter(torch.randn(1, 1, emb_size))
            self.decoder = decoder(emb_size=emb_size, depth=2)

    def random_masking(self, x, mask_ratio=0.5):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
      
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)

        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.zeros([B, L], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x, **kwargs):
        
        # x expected as (B, C, T)
        if x.ndim == 4:  # (B, C, P, Tp) -> (B, C, P*Tp)
            b, c, p, tp = x.shape
            x = x.reshape(b, c, p * tp)
        assert x.ndim == 3, f"Expected (B,C,T) or (B,C,P,Tp), got {tuple(x.shape)}"
        assert x.shape[1] == 45, \
            f"Got C={x.shape[1]}, but PatchEmbedding.num_channels=45. " \
            "Subset/reorder channels or instantiate with the correct num_channels."

        original_x = self.embedding(x) 

        if self.pretrainmode:

            x_masked, mask, ids_restore = self.random_masking(original_x,mask_ratio=self.mask_ratio)

            encoded = self.transformer(x_masked)
 
            reconstructed = self.decoder(encoded)

            cls_output = self.clshead(torch.mean(encoded, dim=1))
            
            return cls_output, original_x, reconstructed, None
        else:
            transformed = self.transformer(original_x)
            pooled = torch.mean(transformed, dim=1)
            cls_output = self.clshead(pooled)
            return cls_output
    def init_from_pretrained(self, pretrained_path, freeze_encoder=False, strict=True):
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        model_dict = self.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        
        self.load_state_dict(model_dict, strict=strict)
        
        if freeze_encoder:
            for name, param in self.named_parameters():
                if 'embedding' in name or 'transformer' in name:
                    param.requires_grad = False
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model")


# import your classes from the file where you defined them
# (if they are in the same file, remove these imports and keep classes above)
# from your_module import PatchEmbedding, MultiHeadAttention, TransformerEncoder, mlm_mask

def apply_lora(model, peft_config):
    """
    Apply LoRA to all supported leaf modules (Linear/Conv/Embedding/MHA)
    except the classification head ('clshead').
    Call this *after* model.init_from_pretrained(...).
    """
    from peft import LoraConfig, get_peft_model, TaskType

    SUPPORTED = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.MultiheadAttention)

    # collect exact names of supported *leaf* modules, excluding clshead
    target_names = []
    for name, mod in model.named_modules():
        if not name or name.startswith("clshead"):
            continue
        # leaf only
        if any(True for _ in mod.children()):
            continue

        # figure out parent module + our own basename
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = model.get_submodule(parent_name) if parent_name else None
        basename = name.split(".")[-1]

        # 1) attention q/k/v/projection (Linear) under MultiHeadAttention only
        if isinstance(parent, MultiHeadAttention) and isinstance(mod, nn.Linear):
            if basename in {"keys", "queries", "values", "projection"}:
                target_names.append(name)
                continue

        # 2) MLP fc1/fc2 (Linear) under FeedForwardBlock at indices 0 and 3
        if isinstance(parent, FeedForwardBlock) and isinstance(mod, nn.Linear):
            if basename in {"0", "3"}:   # fc1 is child '0', fc2 is child '3'
                target_names.append(name)
                continue
            
    print("Applying LoRA to the following modules:", target_names)

    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # generic, safe for custom models
        r=int(peft_config["r"]),
        lora_alpha=int(peft_config["lora_alpha"]),
        lora_dropout=float(peft_config["lora_dropout"]),
        target_modules=target_names,
        modules_to_save=["clshead"],  # keep the head as a regular (non-LoRA) module + save it
    )

    model = get_peft_model(model, cfg)

     # make sure head is trainable
    for n, p in model.named_parameters():
        if n.startswith("clshead."):
            p.requires_grad = True

    # # optionally freeze everything except LoRA params + head
    # if freeze_backbone:
    #     for n, p in model.named_parameters():
    #         if ("lora_" in n) or n.startswith("clshead."):
    #             p.requires_grad = True
    #         else:
    #             p.requires_grad = False

    return model




def make_mirepnet(use_lora=False, lora_config=None
) -> nn.Module:
    """
    High-level builder used by your train.py:

    Example:
        from mirepnet_model import make_mirepnet
        model = make_mirepnet(
            n_classes=2,
            checkpoint_path=None,  # use in-repo default
            use_lora=True,
            freeze_encoder=True,   # train adapters + head only
        )
    """
    
    ckpt = "weights/pretrained-models/MIRepNet.pth"
    model = mlm_mask(
        n_classes=2,
        pretrain=ckpt,
        pretrainmode=False,
        
    )
    # 2) load pretrained weights (if provided or default exists)
    model.init_from_pretrained(str(ckpt))
    

    # 3) optionally add LoRA
    if use_lora:
        
        model = apply_lora(
                    model, peft_config=lora_config
                )
        print(f"Applied LoRA with r={lora_config['r']}, alpha={lora_config['lora_alpha']}, dropout={lora_config['lora_dropout']}.")
        # print(model)
        print("n trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad), "out of ", sum(p.numel() for p in model.parameters()))
    # # 4) freeze as requested
    # if freeze_encoder:
    #     if use_lora:
    #         # freeze all but LoRA + head
    #         freeze_all_but_head_and_lora(model)
    #     else:
    #         # freeze encoder; keep head trainable
    #         for n, p in model.named_parameters():
    #             if n.startswith("clshead."):
    #                 p.requires_grad = True
    #             elif ('embedding' in n) or ('transformer' in n):
    #                 p.requires_grad = False

    return model


if __name__ == "__main__":
    # Simple test
    model = make_mirepnet(use_lora=True, lora_config={'r':2, 'alpha':8, 'dropout':0.1, 'target_modules':["proj"]})
    x = torch.randn(2, 45, 4, 250)  # Example input
    x1 = torch.randn(1, 45, 1000)
    out = model.forward(x)
    print(out.shape)  # Should be (2, n_classes)
    out1 = model.forward(x1)
    print(out1.shape)  # Should be (1, n_classes)