import torch
import torch.nn as nn
from cnn_layer import CNNLayer


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            CNNLayer(num_cnns=2, num_filters=64),
            CNNLayer(num_cnns=2, num_filters=128),
            CNNLayer(num_cnns=3, num_filters=256),
            CNNLayer(num_cnns=3, num_filters=512),
            CNNLayer(num_cnns=3, num_filters=512),
            nn.Flatten(),
            nn.LazyLinear(4096, bias=False),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.LazyLinear(4096, bias=False),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    model = VGG()
    print(model(x).shape)
