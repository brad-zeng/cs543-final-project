import torch
import torch.nn as nn

def gram_matrix(x):
    B, C, H, W = x.shape
    f = x.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)

class GramBlock(nn.Module):
    def __init__(self, C, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(C * C, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        G = gram_matrix(x)
        return self.fc(G.flatten(1))
