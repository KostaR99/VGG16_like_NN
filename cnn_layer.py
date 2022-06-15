from turtle import forward
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, num_filters) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(num_filters, 3, 1, 1, padding_mode='reflect'),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.block(x)


class CNNLayer(nn.Module):
    def __init__(self, num_cnns: int, num_filters: int):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CNNBlock(num_filters=num_filters)
                for _ in range(num_cnns)
            ]
        )
        self.max_pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.max_pool(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = CNNLayer(2, 64)
    print(model(x).shape)
