"""ResNet18 и ResNet50, реализованные с нуля."""

import torch
import torch.nn as nn


class BasicBlock_1(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.first_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False )
        self.first_norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.second_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False )
        self.second_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first_conv2d(x)
        y = self.first_norm(y)
        y = self.act(y)
        y = self.second_conv2d(y)
        y = self.second_norm(y)
        y = x + y
        y = self.act(y)
        return y
    

class BasicBlock_2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.first_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False )
        self.first_norm = nn.BatchNorm2d(in_channels * 2)
        self.second_conv2d = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False )
        self.second_norm = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.ReLU()

        self.skip_connection_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, stride=2, kernel_size=1, padding=0, bias=False)
        self.skip_connection_norm = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first_conv2d(x)
        y = self.first_norm(y)
        y = self.act(y)
        y = self.second_conv2d(y)
        y = self.second_norm(y)

        x = self.skip_connection_conv2d(x)
        x = self.skip_connection_norm(x)
        
        y = x + y
        y = self.act(y)
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv2d = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.firs_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock_1(64),
            BasicBlock_1(64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock_2(64),
            BasicBlock_1(128),
        )   # in 64 out 128, h / 2 x w / 2
        self.layer3 = nn.Sequential(
            BasicBlock_2(128),
            BasicBlock_1(256),
        )   # in 128 out 256, h / 4 x w / 4
        self.layer4 = nn.Sequential(
            BasicBlock_2(256),
            BasicBlock_1(512),
        )    # in 128 out 512, h / 8 x w / 8
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv2d(x)
        x = self.first_norm(x)
        x = self.act(x)
        x = self.firs_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class BottleneckBlock_1(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(in_channels)
        self.conv2d_3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels * 4)
        self.conv2d_skip = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_skip = nn.BatchNorm2d(in_channels * 4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2d_1(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.conv2d_2(y)
        y = self.norm_2(y)
        y = self.act(y)
        y = self.conv2d_3(y)
        y = self.norm_3(y)
        
        x = self.conv2d_skip(x)
        x = self.norm_skip(x)

        return self.act(x + y)
    

class BottleneckBlock_2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out_channels = in_channels // 4
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.conv2d_3 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2d_1(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.conv2d_2(y)
        y = self.norm_2(y)
        y = self.act(y)
        y = self.conv2d_3(y)
        y = self.norm_3(y)

        return self.act(x + y)
    

class BottleneckBlock_3(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(in_channels // 2)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(in_channels // 2)
        self.conv2d_3 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels * 2)
        self.conv2d_skip = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.norm_skip = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2d_1(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.conv2d_2(y)
        y = self.norm_2(y)
        y = self.act(y)
        y = self.conv2d_3(y)
        y = self.norm_3(y)

        x = self.conv2d_skip(x)
        x = self.norm_skip(x)

        return self.act(x + y)


# ---------ResNet50-------
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BottleneckBlock_1(64),
            BottleneckBlock_2(64 * 4),
            BottleneckBlock_2(64 * 4),
            )
        self.layer2 = nn.Sequential(
            BottleneckBlock_3(64 * 4),
            BottleneckBlock_2(128 * 4),
            BottleneckBlock_2(128 * 4),
            BottleneckBlock_2(128 * 4),
        )
        self.layer3 = nn.Sequential(
            BottleneckBlock_3(128 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
        )
        self.layer4 = nn.Sequential(
            BottleneckBlock_3(256 * 4),
            BottleneckBlock_2(512 * 4),
            BottleneckBlock_2(512 * 4),
        )
        self.pool_2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool_1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        y = self.fc(x)
        return y
