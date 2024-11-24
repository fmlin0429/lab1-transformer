import torch
from torch import nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term) # Apply cos to odd indices
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :] # (max_len, batch_size, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, mlp_dimension: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, mlp_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dimension, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        
        assert self.head_dim * heads == dimension, "Dimension must be divisible by heads"
        
        self.w_q = nn.Linear(dimension, dimension, bias=False)
        self.w_k = nn.Linear(dimension, dimension, bias=False)
        self.w_v = nn.Linear(dimension, dimension, bias=False)
        
        self.w_o = nn.Linear(dimension, dimension, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        q_prime = self.w_q(q)
        k_prime = self.w_k(k)
        v_prime = self.w_v(v)
        
        q_prime = self.split_heads(q_prime)
        k_prime = self.split_heads(k_prime)
        v_prime = self.split_heads(v_prime)
        
        x, self.attention_scores = self.attention(q_prime, k_prime, v_prime, mask, self.dropout)
        
        x = self.combine_heads(x)
        
        return self.w_o(x)
        
    def split_heads(self, x):
        batch_size, seq_length, dimension = x.size()
        return x.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2) # (batch_size, heads, seq_length, head_dim)
    
    def attention(self, q_prime, k_prime, v_prime, mask=None, dropout: nn.Module = None):
        d_k = q_prime.size(-1)
        attention_scores = torch.matmul(q_prime, k_prime.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, v_prime), attention_scores
    
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dimension) # (batch_size, seq_length, dimension)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask=src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features: int, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask=None, cross_attention_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask=src_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, mask=cross_attention_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask=None, cross_attention_mask=None):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, cross_attention_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1) # (batch_size, seq_length, vocab_size)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, target_embedding: InputEmbedding, src_positional_encoding: PositionalEncoding, target_positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_positional_encoding = src_positional_encoding
        self.target_positional_encoding = target_positional_encoding
        self.projection_layer = projection_layer
    
    def forward(self, src_seq, target_seq, src_mask, target_mask, src_padding_mask, target_padding_mask):
        src_seq = self.src_embedding(src_seq)
        target_seq = self.target_embedding(target_seq)
        src_seq = self.src_positional_encoding(src_seq)
        target_seq = self.target_positional_encoding(target_seq)
        
        encoder_output = self.encoder(src_seq, src_mask, src_padding_mask)
        decoder_output = self.decoder(target_seq, encoder_output, src_mask, target_mask)
        
        output = self.projection_layer(decoder_output)
        
        return output

    def encode(self, src_seq, src_mask):
        src_seq = src_seq.long()
        src = self.src_embedding(src_seq)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target_seq, target_mask):
        target_seq = target_seq.long()
        tgt = self.target_embedding(target_seq)
        tgt = self.target_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, target_mask)
    
    def project(self, decoder_output):
        return self.projection_layer(decoder_output)
        
    
def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_length: int, target_seq_length: int, d_model: int = 512, N: int = 6, h: int = 8, d_ff: int = 4, dropout: float = 0.1):
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    target_embedding = InputEmbedding(d_model, target_vocab_size)
    
    src_positional_encoding = PositionalEncoding(d_model, src_seq_length)
    target_positional_encoding = PositionalEncoding(d_model, target_seq_length)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff * d_model, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff * d_model, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, target_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_positional_encoding, target_positional_encoding, projection_layer)
    
    # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer