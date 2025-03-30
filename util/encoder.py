import torch.nn as nn
from .attention import MultiHeadAttention
from .feedback import FeedForward, AddAndNorm, LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float):
        super().__init__()
        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(dropout_rate)
        self.addnorm_2 = AddAndNorm(dropout_rate)

    def forward(self, encoder_input, encoder_mask):
        # takes encoder input from skip connection and adding it with the 
        # output of MultiHead attention block
        encoder_input = self.addnorm_1(encoder_input, lambda encoder_input: self.mutlihead_attention(encoder_input, encoder_input, encoder_input, encoder_mask))

        # taking encoder output of MultiHead attention block from skip 
        # connection and adding it with the output of Feedforward layer
        encoder_input = self.addnorm_1(encoder_input, self.feed_forward)
        return encoder_input
    
class Encoder(nn.Module):
    def __init__(self, encoderblocklist: nn.ModuleList):
        super().__init__()
        self.encoderblocklist = encoderblocklist
        self.layer_norm = LayerNorm()

    def forward(self, encoder_input, encoder_mask):
        # looping through all 6 encoder blocks
        for encoderblock in self.encoderblocklist:
            encoder_input = encoderblock(encoder_input, encoder_mask)
        
        # normalize the final encoder block output and return. This encoder
        # output will be used later on as key and value for the cross attention
        # in decoder block
        encoder_output = self.layer_norm(encoder_input)
        return encoder_output
    