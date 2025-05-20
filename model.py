# model.py

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool_layer = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc_layer1 = nn.Linear(32 * 16 * 16, 128)  # 64x64 input -> 32x32 after pool -> 16x16 after 2nd pool
        self.fc_layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool_layer(F.relu(self.conv_layer1(x)))  # (batch, 16, 32, 32)
        x = self.pool_layer(F.relu(self.conv_layer2(x)))  # (batch, 32, 16, 16)
        x = x.view(x.size(0), -1)             # flatten (batch, 32*16*16)
        x = F.relu(self.fc_layer1(x))
        x = self.fc_layer2(x)
        return x
