import torch
import torch.nn as nn

class UNetInpainter(nn.Module):
    def __init__(self, input_channels=2, output_channels=2):
        super(UNetInpainter, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.enc1 = conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256) # 512 because of skip connection
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with Skip Connections
        # We use torch.cat to combine the upsampled features with encoder features
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_seq(x)
        out += residual  # The skip connection
        return self.relu(out)

class ResNetInpainter(nn.Module):
    def __init__(self, input_channels=2, num_blocks=8, internal_channels=64):
        super(ResNetInpainter, self).__init__()
        
        # Initial Feature Extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, internal_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Stack of Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(internal_channels) for _ in range(num_blocks)]
        )
        
        # Final Reconstruction Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels // 2, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: [Batch, 2, L, L]
        out = self.initial_conv(x)
        out = self.res_blocks(out)
        out = self.final_conv(out)
        
        # Global Residual Learning: 
        # Adding the input back ensures we keep the measured sensor data intact
        return out + x