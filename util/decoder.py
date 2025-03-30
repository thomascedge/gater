import torch.nn as nn
from util.attention import MultiHeadAttention
from util.feedback import FeedForward, AddAndNorm, LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self, masked_multihead_attention: MultiHeadAttention, cross_multihead_attention: MultiHeadAttention, feed_forward: FeedForward, droupout_rate: float):
        super().__init()
        self.masked_multihead_attention = masked_multihead_attention
        self.cross_multihead_attention = cross_multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(droupout_rate)
        self.addnorm_2 = AddAndNorm(droupout_rate)
        self.addnorm_3 = AddAndNorm(droupout_rate)

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        # input from skip connection and adding it with the output of Masked
        # Multi-Head attention block
        decoder_input = self.addnorm_1(decoder_input, lambda decoder_input: self.masked_multihead_attention(decoder_input, decoder_input, decoder_mask))

        # takes first output and adding it with the output of MultiHead 
        # attention block
        decoder_input = self.addnorm_2(decoder_input, lambda decoder_input: self.cross_multihead_attention(decoder_input, encoder_output, encoder_mask))

        # takes second ouput and adds with Feedforward layer
        decoder_input = self.addnorm_3(decoder_input, self.feed_forward)
        return decoder_input
    
class Decoder(nn.Module):
    def __init__(self, decoderblocklist: nn.ModuleList):
        super().__init__()
        self.decoderblocklist = decoderblocklist
        self.layer_norm = LayerNorm

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        for decoderblock in self.decoderblocklist:
            decoder_input = decoderblock(decoder_input, encoder_output, encoder_mask, decoder_mask)
        decoder_output = self.layer_norm(decoder_input)
        return decoder_output
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection_layer = nn.layer(d_model, vocab_size)

    def forward(self, decoder_output):
        # projection layer first takes in decoder and feeds it into the linear
        # layer of shape (d_model, vocab_size)
        # Change in shape: decoder_output(batch_size, seq_len, d_model) @ 
        # linear_layer(d_model, vocab_size) => 
        # output(batch_size, seq_len, vocab_size)
        ouptut = self.projection_layer(decoder_output)
        return ouptut
    