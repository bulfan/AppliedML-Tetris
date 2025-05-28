import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, height, width, n_actions):
        super().__init__()
        # a small conv net
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # compute conv output size
        convw = (((width - 4)//2 + 1) - 3)//1 + 1
        convh = (((height - 4)//2 + 1) - 3)//1 + 1
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.unsqueeze(1)        # add channel
        x = x.float() / 7.0       # normalize by max block id
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
