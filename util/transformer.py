import torch.nn as nn
from util.encoder import Encoder
from util.embedding import EmbeddingLayer, PositionalEncoding
from util.decoder import Decoder, ProjectionLayer

class Transformer(nn.module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embed: EmbeddingLayer, 
                 target_embed: EmbeddingLayer, source_pos: PositionalEncoding, 
                 target_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.source_embed = source_embed
        self.source_pos = source_pos
        self.encoder = encoder
        self.target_embed = target_embed
        self.target_pos = target_pos
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, encoder_input, encoder_mask):
        encoder_input = self.source_embed(encoder_input)
        encoder_input = self.source_pos(encoder_input)
        encoder_output = self.encoder(encoder_input, encoder_mask)
        return encoder_output
    
    def decode(self, decoder_output):
        return self.projection_layer(decoder_output)
    