import torch.nn as nn
import torch

class SqueezeExcitationAttentionBlock(nn.Module):
    
    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_channels)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out =  self.avg_pool(x).squeeze(-1).squeeze(-1)
        out = self.fc2(self.relu(self.fc1(out)))
        out = out.unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out) * x


class SpatialAttentionBlock(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(num_channels, 1, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv(x)
        return self.sigmoid(out) * x


class PixelAttentionBlock(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(num_channels, num_channels, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv(x)
        return self.sigmoid(out) * x