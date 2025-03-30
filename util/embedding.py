import torch 
import torch.nn as nn
import math

class EmbeddingLayer(nn.Model):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        # using embedding layer to map token id to embedding vector with shape
        # vocab_size x d_model. vocab_size is the vocabulary size of the 
        # training data created by tokenizer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        embedding_output = self.embedding(input) * math.sqrt(self.d_model)
        return embedding_output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_seq_len, d_model)

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0 / d_model)))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # considering batches of input sentences are expected, the extra
        # dimension caters to batch number needs add in 0th position
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_embedding):
        input_embedding = input_embedding + (self.pe[:, :input_embedding.shape[1]])
        return self.dropout(input_embedding)