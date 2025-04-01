import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()

        # defining dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be dividible by number of heads"

        # new dimension of each self attention heads
        self.d_k = d_model // num_heads

        # weight matrix defined, learnable parameters
        # see Step 5: Muti-head Attention Block on article for explination
        # https://medium.com/towards-artificial-intelligence/build-your-own-large-language-model-llm-from-scratch-using-pytorch-9e9945c24858
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, encoder_mask):
        # model will be trained with batches of sequences, therefore batch_size
        # will be included in the shape query, Key and value are calculated by
        # matrix multiplication of corresponding weights with the input embeddings
        # Change of shape: q(batch_size, seq_len, d_model) 
        # @ W_q(d_model, d_model) => query(batch_size, seq_len, d_model) 
        # [same goes to key and value]
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # dividing query, key, and value into number of heads, hence new 
        # dimension will be d_k. Change of shape: 
        # query(batch_size, seq_len, d_model) => 
        # query(batch_szie, seq_len, num_heads, d_k) ->
        # query(batch_size, num_heads, seq_len, d_k) [same for key and value]
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # :: SELF ATTENTION BLOCK ::

        # attention score calculated to find similarity or relation of query
        # with key of iteself and all other embeddeding in the sequence
        # Change of shape: query(batch_size, num_heads, seq_len, d_k) 
        # @ key (batch_size, num_heads, seq_len, d_k) => 
        # attention_score(batch_size, num_heads, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if mask is provided the attention score needs to be modified as per
        # the mask value
        if encoder_mask is not None:
            attention_score.masked_fill_(encoder_mask==0, -1e9)
        
        # softmax operation calculates the probability distribution among all
        # the attention scores. This will determine which embedding is more
        # similar to the given query embedding and assign the attention weight
        # accordingly
        # Change of shape: same as attention_score
        attention_score = attention_score.softmax(dim=-1)

        if self.dropout is not None:
            attention_score = self.dropout(attention_score)

        # matrix mulitiplication of attention_weight with value embedding.
        # Change of shape: attention_score(batch_size, num_heads, seq_len, seq_len)
        # @ value(batch_size, num_heads, seq_len, d_k) =>
        # attention_outut(batch_size, num_heads, seq_len, d_k)
        attention_output = attention_score @ value

        # :: SELF ATTENTION BLOCK ENDS ::

        # all heads will be concatenated back to a single head
        # Change of shape: attention_output(batch_size, num_heads, seq_len, d_k) =>
        # attention_output(batch_size, seq_len, num_heads, d_k) =>
        # attention_output(batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(attention_output.shape[0], -1, self.num_heads * self.d_k)

        # attention_output is matrix multiplied with output weight matrix to
        # give final multihead attention output. The shape of the 
        # multihead_output is the same as the embedding input
        # Change of shape: attention_output(batch_size, seq_len, d_model) @
        # W_o(d_model, d_model) => multihead_ouptut(batch_size, seq_len, d_model)
        multihead_output = self.W_o(attention_output)

        return multihead_output
    