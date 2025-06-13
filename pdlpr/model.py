import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
#  Basic building blocks
# -----------------------------

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + LeakyReLU (default slope=0.1)."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 1, slope: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """Focus structure from YOLOv5 - spatially slice then concatenate along channel dim."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4, out_channels, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape : B C H W  ->  B 4C H/2 W/2 after slicing & cat
        patch = torch.cat(
            (
                x[..., ::2, ::2],       # top‑left
                x[..., 1::2, ::2],      # top‑right
                x[..., ::2, 1::2],      # bottom‑left
                x[..., 1::2, 1::2],     # bottom‑right
            ),
            dim=1,
        )
        return self.conv(patch)


class ResBlock(nn.Module):
    """Two 3x3 ConvBNAct layers with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = ConvBNAct(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class ConvDownSample(nn.Module):
    """Strided 3x3 conv (s=2) instead of pooling to keep more semantics."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels, out_channels, k=3, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------
#  Improved Global Feature Extractor (IGFE)
# -----------------------------

class IGFE(nn.Module):
    """Backbone that downsamples 48x144 input to a 6x18 feature map with 512 channels."""

    def __init__(self, in_channels: int = 3, channels = (64, 128, 256, 512)):
        super().__init__()
        self.focus = Focus(in_channels, channels[0])           # 24x72

        self.down1 = ConvDownSample(channels[0], channels[1]) # 12x36
        self.res1a = ResBlock(channels[1])
        self.res1b = ResBlock(channels[1])

        self.down2 = ConvDownSample(channels[1], channels[2]) # 6x18
        self.res2a = ResBlock(channels[2])
        self.res2b = ResBlock(channels[2])

        # one more stage to hit 6x18 with 512 ch
        self.down3 = ConvDownSample(channels[2], channels[3]) # 3x9
        self.res3a = ResBlock(channels[3])
        self.res3b = ResBlock(channels[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1a(x); x = self.res1b(x)

        x = self.down2(x)
        x = self.res2a(x); x = self.res2b(x)

        x = self.down3(x)
        x = self.res3a(x); x = self.res3b(x)
        return x  # B 512 3 9  (will be flattened later)


# -----------------------------
#  Positional encodings
# -----------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (can be shared for encoder / decoder)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x max_len x d_model
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding (x: B x N x d)."""
        return x + self.pe[:, : x.size(1)].clone().detach()


# -----------------------------
#  Transformer Encoder / Decoder blocks
# -----------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention
        attn, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + attn)
        # FFN
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn, _ = self.self_mha(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)
        tgt = self.norm1(tgt + self_attn)

        cross_attn, _ = self.cross_mha(tgt, memory, memory, need_weights=False)
        tgt = self.norm2(tgt + cross_attn)

        ff = self.ffn(tgt)
        tgt = self.norm3(tgt + ff)
        return tgt


# utility ---------------------------------------------------------------------

def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out (set True) the upper-triangular part to prevent attending to future positions."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


# -----------------------------
#  Full PDLPR network
# -----------------------------

class PDLPR(nn.Module):
    """PDLPR - Parallel Decoder License Plate Recognition network."""

    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_ff: int = 512,
        max_len: int = 18,
    ):
        super().__init__()
        self.backbone = IGFE()
        self.flatten = nn.Flatten(2)  # keep B & C, flatten (H, W)

        # Positional encodings
        self.pos_enc = PositionalEncoding(d_model, max_len=108)  # 3 x 9 == 27 tokens; original paper uses 6x18=108
        self.pos_dec = PositionalEncoding(d_model, max_len=max_len + 1)

        # Transformer stacks
        self.enc_layers = nn.ModuleList(
            [EncoderBlock(d_model, nhead, dim_ff) for _ in range(num_encoder_layers)]
        )
        self.embedding = nn.Embedding(num_classes, d_model)
        self.dec_layers = nn.ModuleList(
            [DecoderBlock(d_model, nhead, dim_ff) for _ in range(num_decoder_layers)]
        )

        self.classifier = nn.Linear(d_model, num_classes) 
        self.ctc_head   = nn.Linear(d_model, num_classes)   
        self.max_len = max_len
        self.num_classes = num_classes

    # ---------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # backbone -> (B, 512, H=3, W=9)   (for 48x144 input)
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        feat = self.flatten(feat)  # B, C, H*W
        feat = feat.transpose(1, 2)  # B, N, C  (N = H*W)
        feat = self.pos_enc(feat)
        for layer in self.enc_layers:
            feat = layer(feat)
        return feat  # B, N, C

    # ---------------------------------------------------------------------
    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """Forward for training.

        Args:
            images: B x 3 x 48 x 144 raw plate crops.
            tgt_seq: B x L int64 tensor - **must include <sos> token at position 0**, but **not include <eos>**.
        Returns:
            logits: B x L x num_classes - unnormalized scores for CTC / Cross-Entropy.
        """
        memory = self.encode(images)

        tgt = self.embedding(tgt_seq)  # B L C
        tgt = self.pos_dec(tgt)
        mask = subsequent_mask(tgt.size(1)).to(tgt.device)

        for layer in self.dec_layers:
            tgt = layer(tgt, memory, tgt_mask=mask)

        logits = self.classifier(tgt)  # B L num_classes
        return logits
    
    def forward_ctc(self, images: torch.Tensor) -> torch.Tensor:
        """Run backbone+encoder and project to vocab logits for CTC."""
        memory = self.encode(images)                  # B, N(=T), C
        logits  = self.ctc_head(memory)               # B, N, V
        return logits.permute(1, 0, 2)                # T, B, V  (time major)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, images: torch.Tensor, sos_id: int, eos_id: int, device: Optional[torch.device] = None):
        """Greedy decode (no beam‑search). Returns list[int] per sample (without SOS)."""
        device = device or images.device
        memory = self.encode(images.to(device))
        B = images.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(self.max_len):
            tgt = self.embedding(ys)
            tgt = self.pos_dec(tgt)
            mask = subsequent_mask(tgt.size(1)).to(device)
            out = tgt
            for layer in self.dec_layers:
                out = layer(out, memory, tgt_mask=mask)
            logits = self.classifier(out[:, -1])  # last step logits, shape B x num_classes
            next_token = logits.argmax(-1, keepdim=True)  # greedy
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return ys[:, 1:]  # strip SOS token


__all__ = ["PDLPR"]