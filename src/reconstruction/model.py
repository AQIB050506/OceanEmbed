"""
Attention-Enhanced 3D U-Net++ for Ocean Subsurface Temperature Reconstruction.

Architecture based on:
- Wang et al. (2026): "Attention enhanced 3D-U-Net++ ocean temperature and salinity reconstruction"
- TS-Cast (Chae et al., 2026): FiLM conditioning on climatological priors

Key features:
- 3D convolutions for spatial-temporal feature extraction
- CBAM (Convolutional Block Attention Module) for depth-aware feature gating
- Encoder-decoder with skip connections
- Optional FiLM conditioning on climatological profiles
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.config import MODEL_CONFIG


class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class DoubleConv3DBlock(nn.Module):
    """Two consecutive 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block with double conv, CBAM attention, and max pooling."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv_block = DoubleConv3DBlock(in_ch, out_ch, dropout)
        self.cbam = CBAM(out_ch)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.cbam(self.conv_block(x))
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_block = DoubleConv3DBlock(in_ch // 2 + skip_ch, out_ch, dropout)
        self.cbam = CBAM(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.cbam(self.conv_block(x))
        return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for climatological conditioning."""

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class AttentionUNet3D(nn.Module):
    """
    Attention-Enhanced 3D U-Net++ for subsurface temperature reconstruction.

    Input: Multi-channel surface data (B, C_in, T, H, W)
           where C_in = number of surface variables (SST, SSH, SSS, Wind_U, Wind_V)
    Output: Temperature profiles at N depth levels (B, N_depth, H, W)
    """

    def __init__(
        self,
        input_channels: int = None,
        n_depth_levels: int = None,
        embed_dim: int = None,
        dropout: float = None,
        use_film: bool = True,
    ):
        super().__init__()
        cfg = MODEL_CONFIG
        input_channels = input_channels or cfg["input_channels"]
        n_depth_levels = n_depth_levels or cfg["n_depth_levels"]
        embed_dim = embed_dim or cfg["embed_dim"]
        dropout = dropout or cfg["dropout"]

        self.use_film = use_film
        self.n_depth_levels = n_depth_levels

        self.input_conv = nn.Conv3d(input_channels, embed_dim, kernel_size=1)

        self.enc1 = EncoderBlock(embed_dim, embed_dim, dropout)
        self.enc2 = EncoderBlock(embed_dim, embed_dim * 2, dropout)
        self.enc3 = EncoderBlock(embed_dim * 2, embed_dim * 4, dropout)

        self.bottleneck = DoubleConv3DBlock(embed_dim * 4, embed_dim * 4, dropout)
        self.bottleneck_cbam = CBAM(embed_dim * 4)

        self.dec3 = DecoderBlock(embed_dim * 4, embed_dim * 4, embed_dim * 2, dropout)
        self.dec2 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, dropout)
        self.dec1 = DecoderBlock(embed_dim, embed_dim, embed_dim, dropout)

        self.depth_projection = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, n_depth_levels, kernel_size=1),
        )

        if use_film:
            self.film = FiLMLayer(input_channels, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, climatology: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Surface input tensor (B, C, T, H, W)
            climatology: Optional climatological profile for FiLM conditioning

        Returns:
            Reconstructed temperature profiles (B, N_depth, H, W)
        """
        x = self.input_conv(x)

        if self.use_film and climatology is not None:
            cond = climatology.mean(dim=(2, 3, 4))
            x = self.film(x, cond)

        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)

        x = self.bottleneck_cbam(self.bottleneck(x))

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = torch.mean(x, dim=2)

        output = self.depth_projection(x)

        return output


def build_model(device: str = "cuda") -> AttentionUNet3D:
    """Build and return the model."""
    model = AttentionUNet3D()
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
