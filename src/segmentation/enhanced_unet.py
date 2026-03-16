"""
Enhanced 3D U-Net with Deep Supervision and Improved Architecture
Optimized for small brain metastasis detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Double convolution block with residual connection
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.15):
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
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )

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


class AttentionGate3D(nn.Module):
    """Additive Attention Gate for 3D U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DownBlock(nn.Module):
    """Downsampling with residual conv block"""
    def __init__(self, in_channels, out_channels, dropout_p=0.15):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling with attention and residual conv"""
    def __init__(self, in_channels, out_channels, dropout_p=0.15, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate3D(
                F_g=out_channels,
                F_l=out_channels,
                F_int=out_channels // 2
            )

        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        # Apply attention to skip connection
        if self.use_attention:
            skip = self.attention(g=x, x=skip)

        # Concatenate
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DeepSupervisedUNet3D(nn.Module):
    """
    3D U-Net with Deep Supervision for small lesion detection

    Features:
    - Residual connections throughout
    - Attention gates in decoder
    - Deep supervision at multiple scales
    - Optimized for small metastases

    Args:
        in_channels: Number of input modalities
        out_channels: Number of output classes
        base_channels: Base number of channels
        depth: Number of downsampling levels
        dropout_p: Dropout probability
        deep_supervision: Enable deep supervision outputs
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=4,
        dropout_p=0.15,
        deep_supervision=True
    ):
        super().__init__()

        self.depth = depth
        self.deep_supervision = deep_supervision

        # Channel progression
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Initial convolution
        self.inc = ResidualConvBlock(in_channels, channels[0], dropout_p)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p)
            for i in range(depth)
        ])

        # Bottleneck
        self.bottleneck = ResidualConvBlock(channels[depth], channels[depth], dropout_p)

        # Decoder (upsampling path) with attention
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout_p, use_attention=True)
            for i in range(depth-1, -1, -1)
        ])

        # Final output head
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

        # Deep supervision heads (output at multiple scales)
        # After upsampling, channels go from channels[depth-1] down to channels[0]
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(channels[depth-1-i], out_channels, kernel_size=1)
                for i in range(depth)
            ])

    def forward(self, x, return_ds=None):
        """
        Forward pass

        Args:
            x: Input tensor
            return_ds: If True, return deep supervision outputs during training

        Returns:
            If deep_supervision and return_ds:
                [main_output, ds_output_1, ds_output_2, ...]
            Else:
                main_output
        """
        # Override with instance setting if not specified
        if return_ds is None:
            return_ds = self.deep_supervision

        # Encoder
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)

        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with deep supervision
        ds_outputs = []

        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i+2)]  # Get corresponding skip connection
            x = up(x, skip)

            # Deep supervision output at this scale
            if self.deep_supervision and return_ds and i < len(self.ds_heads):
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)

        # Final output
        main_output = self.outc(x)

        if self.deep_supervision and return_ds:
            return [main_output] + ds_outputs
        else:
            return main_output


class HybridUNet3D(nn.Module):
    """
    Hybrid U-Net combining multiple architectural improvements

    Features:
    - Residual connections for better gradient flow
    - Attention gates for feature selection
    - Deep supervision for multi-scale learning
    - Squeeze-and-Excitation blocks for channel attention
    - Optimized for very small lesion detection

    Args:
        in_channels: Number of input modalities
        out_channels: Number of output classes
        base_channels: Base feature channels
        depth: Network depth
        dropout_p: Dropout rate
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=4,
        dropout_p=0.15
    ):
        super().__init__()

        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder
        self.inc = ResidualConvBlock(in_channels, channels[0], dropout_p)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p)
            for i in range(depth)
        ])

        # Bottleneck with SE block
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(channels[depth], channels[depth], dropout_p),
            SqueezeExcitation3D(channels[depth])
        )

        # Decoder
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout_p, use_attention=True)
            for i in range(depth-1, -1, -1)
        ])

        # Output
        self.outc = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[0], out_channels, kernel_size=1)
        )

        # Deep supervision
        # After upsampling, channels go from channels[depth-1] down to channels[0]
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(channels[depth-1-i], out_channels, kernel_size=1)
            for i in range(depth)
        ])

    def forward(self, x, return_ds=True):
        # Encoder
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)

        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        ds_outputs = []
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i+2)]
            x = up(x, skip)

            if return_ds and i < len(self.ds_heads):
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)

        # Final
        main_output = self.outc(x)

        if return_ds and self.training:
            return [main_output] + ds_outputs
        else:
            return main_output


class SqueezeExcitation3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D
    Learns channel-wise attention
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing DeepSupervisedUNet3D...")
    model = DeepSupervisedUNet3D(
        in_channels=4,
        base_channels=16,
        depth=4,
        deep_supervision=True
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(1, 4, 128, 128, 128).to(device)

    with torch.no_grad():
        outputs = model(x, return_ds=True)
        print(f"Number of outputs: {len(outputs)}")
        print(f"Main output shape: {outputs[0].shape}")
        for i, ds_out in enumerate(outputs[1:]):
            print(f"DS output {i+1} shape: {ds_out.shape}")

    print("\nTesting HybridUNet3D...")
    model2 = HybridUNet3D(
        in_channels=4,
        base_channels=16,
        depth=4
    ).to(device)

    print(f"Parameters: {count_parameters(model2):,}")

    del model, model2
    torch.cuda.empty_cache()
    print("\nModels tested successfully!")
