"""U-Net-энкодер-декодер, приспособленный под классификацию."""

import torch
import torch.nn as nn


class UNetForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder_1 = nn.Sequential( # 256x256 -> 256x256
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
        )

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2) # 256x256 -> 128x128

        self.encoder_2 = nn.Sequential( # 128x128 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) # 128x128 -> 64x64

        self.encoder_3 = nn.Sequential( # 64x64 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2) # 64x64 -> 32x32

        self.botneck = nn.Sequential( # 32x32 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 32x32 -> 64x64
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 256 = 128(upconv) + 128(skip), 64x64 -> 64x64
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 64x64 -> 64x64
            nn.ReLU(),
        )
        
        self.upconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 64x64 -> 128x128
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 128 = 64(upconv) + 64(skip), 128x128 -> 128x128
            nn.ReLU(),
            nn.BatchNorm2d(64),            
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 128x128 -> 128x128
            nn.ReLU(),
        )
        
        self.upconv_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # 128x128 -> 256x256
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # 64 = 32(upconv) + 32(skip), 256x256 -> 256x256
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 256x256 -> 256x256
            nn.ReLU(),
        )
         
        self.pool = nn.AdaptiveAvgPool2d(1) # [bacth, 32, 1, 1]
        self.output = nn.Linear(32, 2) # (real/fake)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Выходы энкодеров, сохраняем для skip-connections
        encoder_1_output = self.encoder_1(x)                      # [batch, 32, 256, 256]
        max_pool_1_output = self.max_pool_1(encoder_1_output)     # [batch, 32, 128, 128]

        encoder_2_output = self.encoder_2(max_pool_1_output)      # [batch, 64, 128, 128]
        max_pool_2_output = self.max_pool_2(encoder_2_output)     # [batch, 64, 64, 64]

        encoder_3_output = self.encoder_3(max_pool_2_output)      # [batch, 128, 64, 64]
        max_pool_3_output = self.max_pool_3(encoder_3_output)     # [batch, 128, 32, 32]

        # Боттлнек (самый глубокий слой)
        botneck_output = self.botneck(max_pool_3_output)          # [batch, 256, 32, 32]

        # Декодеры со skip-connections
        # Уровень 1 декодера
        upconv_3_output = self.upconv_3(botneck_output)           # [batch, 128, 64, 64]
        decoder_3_input = torch.cat([upconv_3_output, encoder_3_output], dim=1)  # [batch, 256, 64, 64]
        decoder_3_output = self.decoder_3(decoder_3_input)        # [batch, 128, 64, 64]
        
        # Уровень 2 декодера
        upconv_2_output = self.upconv_2(decoder_3_output)         # [batch, 64, 128, 128]
        decoder_2_input = torch.cat([upconv_2_output, encoder_2_output], dim=1)  # [batch, 128, 128, 128]
        decoder_2_output = self.decoder_2(decoder_2_input)        # [batch, 64, 128, 128]

        # Уровень 3 декодера
        upconv_1_output = self.upconv_1(decoder_2_output)         # [batch, 32, 256, 256]
        decoder_1_input = torch.cat([upconv_1_output, encoder_1_output], dim=1)  # [batch, 64, 256, 256]
        decoder_1_output = self.decoder_1(decoder_1_input)        # [batch, 32, 256, 256]

        # Выход
        pooled = self.pool(decoder_1_output)                      # [batch, 32, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)               # [batch, 32]
        output = self.output(flattened)                           # [bacth, 2]
        
        return output
