"""Inception V1 и V3, реализованные с нуля."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = False
                 ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = self.relu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_1x1: int,
                 reduce_3x3: int,
                 out_3x3: int,
                 reduce_5x5: int,
                 out_5x5: int,
                 out_1x1_pooling: int
                 ):
        super().__init__()

        # 1x1 conv
        self.branch_1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)

        # 1x1 conv -> 3x3 conv
        self.branch_2 = nn.Sequential(ConvBlock(
            in_channels, reduce_3x3, 1, 1, 0), ConvBlock(reduce_3x3, out_3x3, 3, 1, 1))

        # 1x1 conv -> 5x5 conv
        self.branch_3 = nn.Sequential(ConvBlock(
            in_channels, reduce_5x5, 1, 1, 0), ConvBlock(reduce_5x5, out_5x5, 5, 1, 2))

        # 3x3 maxpool -> 1x1 conv
        self.branch_4 = nn.Sequential(nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1), ConvBlock(in_channels, out_1x1_pooling, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], dim=1)


class InceptionV1(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0), ConvBlock(64, 192, 3, 1, 1))
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3_a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3_b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4_a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4_b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4_c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4_d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4_e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5_a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5_b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        # print('conv_1',x.shape)
        x = self.maxpool_1(x)
        # print('maxpool_1',x.shape)

        x = self.conv_2(x)
        # print('conv_2',x.shape)
        x = self.maxpool_2(x)
        # print('maxpool_2',x.shape)

        x = self.inception_3_a(x)
        # print('3_a',x.shape)

        x = self.inception_3_b(x)
        # print('3_b',x.shape)

        x = self.maxpool_3(x)
        # print('3_b_max',x.shape)

        x = self.inception_4_a(x)
        # print('4_a',x.shape)

        x = self.inception_4_b(x)
        # print('4_b',x.shape)

        x = self.inception_4_c(x)
        # print('4_c',x.shape)

        x = self.inception_4_d(x)
        # print('4_d',x.shape)

        x = self.inception_4_e(x)
        # print('4_e',x.shape)

        x = self.maxpool_4(x)
        # print('maxpool',x.shape)

        x = self.inception_5_a(x)
        # print('5_a',x.shape)

        x = self.inception_5_b(x)
        # print('5_b',x.shape)

        x = self.avgpool(x)
        # print('AvgPool',x.shape)

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_channels: int):
        super().__init__()
        
        self.branch_1 = ConvBlock(in_channels, 64, 1, 1, 0)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 48, 1, 1, 0),
            ConvBlock(48, 64, 5, 1, 2),
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 1),
            ConvBlock(96, 96, 3, 1, 1),
        )

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, pool_channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
            self.branch_4(x)
        ], dim=1)

        
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch_1 = ConvBlock(in_channels, 192, 1, 1, 0)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, 1, 0),
            ConvBlock(128, 128, (1, 7), 1, (0, 3)),
            ConvBlock(128, 192, (7, 1), 1, (3, 0)),
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, 1, 0),
            ConvBlock(128, 128, (7, 1), 1, (3, 0)),
            ConvBlock(128, 128, (1, 7), 1, (0, 3)),
            ConvBlock(128, 128, (7, 1), 1, (3, 0)),
            ConvBlock(128, 192, (1, 7), 1, (0, 3)),
        )

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, 1, 1, 0),
        )
        
    def forward(self, x):
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
            self.branch_4(x),
        ], dim=1)
        
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch_1 = ConvBlock(in_channels, 320, 1, 1, 0)

        self.branch_2_1 = ConvBlock(in_channels, 384, 1, 1, 0)
        self.branch_2_2a = ConvBlock(384, 384, (1, 3), 1, (0, 1))
        self.branch_2_2b = ConvBlock(384, 384, (3, 1), 1, (1, 0))

        self.branch_3_1 = nn.Sequential(
            ConvBlock(in_channels, 448, 1, 1, 0),
            ConvBlock(448, 384, 3, 1, 1),
        )
        self.branch_3_2a = ConvBlock(384, 384, (1, 3), 1, (0, 1))
        self.branch_3_2b = ConvBlock(384, 384, (3, 1), 1, (1, 0))

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, 1, 1, 0),
        )

    def forward(self, x):
        b1 = self.branch_1(x)

        b2 = self.branch_2_1(x)
        b2 = torch.cat([self.branch_2_2a(b2), self.branch_2_2b(b2)], dim=1)

        b3 = self.branch_3_1(x)
        b3 = torch.cat([self.branch_3_2a(b3), self.branch_3_2b(b3)], dim=1)

        b4 = self.branch_4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)
    
    
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch_1 = ConvBlock(in_channels, 384, 3, 2, 0)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 1),
            ConvBlock(96, 96, 3, 2, 0),
        )

        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
        ], dim=1)
        
        
class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch_1 = nn.Sequential(
            ConvBlock(in_channels, 192, 1, 1, 0),
            ConvBlock(192, 320, 3, 2, 0),
        )

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 192, 1, 1, 0),
            ConvBlock(192, 192, (1, 7), 1, (0, 3)),
            ConvBlock(192, 192, (7, 1), 1, (3, 0)),
            ConvBlock(192, 192, 3, 2, 0),
        )

        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
        ], dim=1)
        

class InceptionV3(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, 32, 3, 2, 0)
        self.conv_2 = ConvBlock(32, 32, 3, 1, 0)
        self.conv_3 = ConvBlock(32, 64, 3, 1, 1)
        self.maxpool_1 = nn.MaxPool2d(3, stride=2)

        self.conv_4 = ConvBlock(64, 80, 1, 1, 0)
        self.conv_5 = ConvBlock(80, 192, 3, 1, 0)
        self.maxpool_2 = nn.MaxPool2d(3, stride=2)

        # -------- 35×35 --------
        self.inception_3_a = InceptionA(192, 32) 
        self.inception_3_b = InceptionA(256, 64)
        self.inception_3_c = InceptionA(288, 64)

        # -------- 35->17 --------
        self.reduction_a = ReductionA(288)

        # -------- 17×17 --------
        self.inception_4_a = InceptionB(768)
        self.inception_4_b = InceptionB(768)
        self.inception_4_c = InceptionB(768)
        self.inception_4_d = InceptionB(768)

        # -------- 17->8 --------
        self.reduction_b = ReductionB(768)

        # -------- 8×8 --------
        self.inception_5_a = InceptionC(1280)
        self.inception_5_b = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool_1(x)

        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.maxpool_2(x)

        x = self.inception_3_a(x)
        x = self.inception_3_b(x)
        x = self.inception_3_c(x)

        x = self.reduction_a(x)

        x = self.inception_4_a(x)
        x = self.inception_4_b(x)
        x = self.inception_4_c(x)
        x = self.inception_4_d(x)

        x = self.reduction_b(x)

        x = self.inception_5_a(x)
        x = self.inception_5_b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
