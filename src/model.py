# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentNet(nn.Module):
    """
    Simple feed-forward ANN for tabular regression.
    Must match the architecture used when training (or at least be compatible
    with the saved state_dict).
    """
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

        # weight init (same as training)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x.squeeze(-1)
