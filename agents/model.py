import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Deep Q-Network for Tetris"""
    def __init__(self, state_size, action_size, hidden_layers=[512, 256, 128], dropout_rate=0.2):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)
