"""
Lightweight 3D U-Net for brain metastasis segmentation
Optimized for consumer GPUs with limited VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv3d -> BatchNorm -> ReLU -> Conv3d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResidualConvBlock(nn.Module):
    """
    Double convolution block with residual connection
    Conv3d -> BatchNorm -> ReLU -> Dropout -> Conv3d -> BatchNorm -> + Residual -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 conv for residual if channels don't match
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Add residual
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        out += residual
        out = self.relu(out)

        return out


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class AttentionGate3D(nn.Module):
    """
    Additive Attention Gate for 3D U-Net

    Learns to focus on salient features from skip connections
    using gating signals from decoder path

    Args:
        F_g: channels in gating signal (from decoder)
        F_l: channels in skip connection (from encoder)
        F_int: intermediate channel dimension
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        # Combine and generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal from coarser scale (decoder)
            x: skip connection from encoder
        Returns:
            attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Element-wise multiplication
        return x * psi


class UpBlock(nn.Module):
    """Upsampling block: TransposeConv -> Concat -> ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)  # in_channels because of concat

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch due to padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUpBlock(nn.Module):
    """Upsampling block with attention gate: TransposeConv -> Attention -> Concat -> ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        # Attention gate: F_g=out_channels (after upsample), F_l=out_channels (skip)
        self.attention = AttentionGate3D(
            F_g=out_channels,
            F_l=out_channels,
            F_int=out_channels // 2
        )

        self.conv = ConvBlock(in_channels, out_channels, dropout_p)  # in_channels because of concat

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch due to padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        # Apply attention to skip connection
        skip_att = self.attention(g=x, x=skip)

        # Concatenate
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for brain metastasis segmentation

    Architecture designed for consumer GPUs:
    - Reduced channel dimensions
    - Only 3 levels deep
    - Dropout for regularization
    - Efficient memory usage
    - Optional attention gates and residual connections

    Args:
        in_channels: Number of input modalities (default: 4 for t1_pre, t1_gd, flair, bravo)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of feature channels (default: 16)
        depth: Number of downsampling levels (default: 3)
        dropout_p: Dropout probability (default: 0.1)
        use_attention: Use attention gates in decoder (default: False)
        use_residual: Use residual connections in conv blocks (default: False)
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=3,
        dropout_p=0.1,
        use_attention=False,
        use_residual=False
    ):
        super().__init__()

        self.depth = depth
        self.use_attention = use_attention
        self.use_residual = use_residual
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Select conv block type
        ConvBlockType = ResidualConvBlock if use_residual else ConvBlock

        # Initial convolution
        self.inc = ConvBlockType(in_channels, channels[0], dropout_p)

        # Encoder (downsampling path) - still uses DownBlock but with appropriate ConvBlock inside
        # Note: DownBlock uses ConvBlock directly, so we need to handle this
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p)
            for i in range(depth)
        ])

        # Decoder (upsampling path) - use attention if enabled
        UpBlockType = AttentionUpBlock if use_attention else UpBlock
        self.up_blocks = nn.ModuleList([
            UpBlockType(channels[i+1], channels[i], dropout_p)
            for i in range(depth-1, -1, -1)
        ])

        # Final convolution
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]

        x = x1
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        # Decoder (skip last skip connection, it's the bottleneck)
        skip_connections = skip_connections[:-1][::-1]

        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[i])

        # Output
        x = self.outc(x)
        return x


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Handles class imbalance better than BCE for medical imaging
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE Loss
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling class imbalance
    Better than Dice for small metastases detection

    Args:
        alpha: Weight for false positives (default: 0.7)
        beta: Weight for false negatives (default: 0.3)
        gamma: Focal parameter (default: 4/3)
        smooth: Smoothing constant (default: 1.0)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        focal_tversky = (1 - tversky_index) ** self.gamma

        return focal_tversky


class EnhancedCombinedLoss(nn.Module):
    """
    Multi-component loss for brain metastasis segmentation

    Components:
    1. Dice Loss (region overlap)
    2. Focal Tversky Loss (class imbalance)
    3. BCE Loss (pixel-wise classification)

    Args:
        dice_weight: Weight for Dice loss (default: 0.4)
        focal_tversky_weight: Weight for Focal Tversky loss (default: 0.4)
        bce_weight: Weight for BCE loss (default: 0.2)
    """
    def __init__(self, dice_weight=0.4, focal_tversky_weight=0.4, bce_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal_tversky = self.focal_tversky_loss(pred, target)
        bce = self.bce_loss(pred, target)

        total_loss = (
            self.dice_weight * dice +
            self.focal_tversky_weight * focal_tversky +
            self.bce_weight * bce
        )

        return total_loss


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing LightweightUNet3D...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=3
    ).to(device)

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass with dummy data
    batch_size = 2
    patch_size = (96, 96, 96)
    x = torch.randn(batch_size, 4, *patch_size).to(device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test loss
    target = torch.randint(0, 2, (batch_size, 1, *patch_size)).float().to(device)
    loss_fn = CombinedLoss()
    loss = loss_fn(output, target)
    print(f"\nLoss: {loss.item():.4f}")

    print("\n✓ Model test passed!")
