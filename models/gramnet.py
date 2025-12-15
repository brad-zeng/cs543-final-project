import torch
import torch.nn as nn
from torchvision.models import resnet18
from .gram import GramBlock

class GramNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.g1 = GramBlock(64)
        self.g2 = GramBlock(128)
        self.g3 = GramBlock(256)
        self.g4 = GramBlock(512)

        self.classifier = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        g = torch.cat([
            self.g1(f1),
            self.g2(f2),
            self.g3(f3),
            self.g4(f4)
        ], dim=1)

        return self.classifier(g)
