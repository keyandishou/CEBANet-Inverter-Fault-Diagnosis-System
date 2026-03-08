import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA) Module"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # Adaptive calculation of kernel size
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        y = self.avg_pool(x).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2)
        return x * self.sigmoid(y).expand_as(x)


class Attention(nn.Module):
    """Temporal Attention Mechanism"""
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [Batch, SeqLen, HiddenDim]
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class CNN_LSTM_Attention(nn.Module):
    """
    CEBANet: 1D CNN-BiLSTM-Attention-ECA Network 
    for Inverter Fault Diagnosis
    """
    def __init__(self, num_classes=19):
        super(CNN_LSTM_Attention, self).__init__()
        
        # 1D Spatial Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            ECALayer(64),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            ECALayer(128),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            ECALayer(256),
            
            # Block 4
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            ECALayer(128),
        )
        
        # Temporal Feature Extractor & Classifier
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.attention = Attention(128)  # 64 hidden_size * 2 (bidirectional)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. Spatial representation
        x = self.cnn(x)                       # [B, Channels, SeqLen]
        x = x.permute(0, 2, 1)                # [B, SeqLen, Channels]
        
        # 2. Temporal modeling
        lstm_out, _ = self.lstm(x)            # [B, SeqLen, 128]
        
        # 3. Attention weighting & Classification
        attn_out = self.attention(lstm_out)   # [B, 128]
        return self.fc(attn_out)              # [B, num_classes]