import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ToeplitzUNet(nn.Module):
    def __init__(self, input_channels=2):
        super(ToeplitzUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = UNetBlock(input_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(64, 128)
        
        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64) # 128 because of skip connection
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32) # 64 because of skip connection
        
        # Final Output Layer
        self.final = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder with Skip Connections
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1) # Concatenate along channel dim
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
    


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # The core "Residual" logic: Identity + learned mapping
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)

class ToeplitzResNet(nn.Module):
    def __init__(self, num_blocks=6, input_channels=2):
        super(ToeplitzResNet, self).__init__()
        
        # Initial Feature Extraction
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Series of Residual Blocks (keeping spatial dimensions constant)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*blocks)
        
        # Final Reconstruction Layers
        self.reconstruct = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1) # Mapping back to [Real, Imag]
        )

    def forward(self, x):
        # Store input for a global skip connection (Optional but powerful)
        identity = x 
        
        out = self.initial(x)
        out = self.res_blocks(out)
        out = self.reconstruct(out)
        
        # Final output is the input + learned adjustment
        return out + identity
    


# Quick Test
if __name__ == "__main__":
    model = ToeplitzUNet()
    sample_input = torch.randn(8, 2, 16, 16) # Batch size 8
    output = model(sample_input)
    print(f"Input Shape: {sample_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be (8, 2, 16, 16)