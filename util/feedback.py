import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forawrd(self, input):
        return self.layer_2(self.dropout(torch.relu(self.layer_1(input))))
    
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(512))
        self.beta = nn.Parameter(torch.zeros(512))

    def foward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepim=True)
        return self.gamma * (input - mean) / (std + self.eps) + self.beta
    
class AddAndNorm(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm

    def forward(self, input, sub_layer):
        return input + self.dropout(sub_layer(self.layer_norm(input)))
    