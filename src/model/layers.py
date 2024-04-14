import torch
import torch.nn as nn
from src.model.util import softmax

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model)) # gain

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()

    def forward(self, x):
        x = self.gelu(self.w1(x))
        x = self.w2(x)
        return x
    
def scaledDotProductAttention(q, k, v, mask=None, pdropout=0):
    # k, q of shape (batch_size, ..., seq_len, d_k)
    # v of shape (batch_size, ..., seq_len, d_v)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-1,-2)) / d_k**0.5 # shape (batch_size, ..., seq_len, seq_len)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e15) # fill with -inf whereever mask is False
    # attention = torch.nn.functional.softmax(scores, dim=-1) # shape (batch_size, ..., seq_len, seq_len)
    attention = softmax(scores, dim=-1)
    attention = torch.nn.functional.dropout(attention, pdropout)
    output = attention @ v # shape (batch_size, ..., seq_len, d_v)
    return output

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_pdrop=0):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_v = d_model // num_heads
#         self.d_k = self.d_v

#         self.W_q = nn.Parameter(torch.randn(self.num_heads, self.d_model, self.d_k)) # TODO initialize properly
#         self.W_k = nn.Parameter(torch.randn(self.num_heads, self.d_model, self.d_k)) # TODO init random seed
#         self.W_v = nn.Parameter(torch.randn(self.num_heads, self.d_model, self.d_v))
#         self.W_o = nn.Parameter(torch.randn(num_heads * self.d_v, self.d_model))
#         self.attn_pdrop = attn_pdrop

#     def forward(self, x):
#         # x of shape (batch_size, seq_len, d_model)
#         batch_size = x.size(0)
#         seq_len = x.size(1)
#         x = x.unsqueeze(1) # (batch_size, 1, seq_len, d_model)
#         q = x @ self.W_q # (batch_size, num_heads, seq_len, d_k)
#         k = x @ self.W_k
#         v = x @ self.W_v

#         print(q[0,0])
#         mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() # TODO make a buffer
#         x = scaledDotProductAttention(q, k, v, mask, self.attn_pdrop) # shape (batch_size, num_heads, seq_len, d_v)

#         print(x[0,0])
#         x = x.transpose(1, 2) # shape (batch_size,seq_len, num_heads, d_v)
#         x = x.reshape(batch_size, seq_len, self.num_heads * self.d_v)
#         x = x @ self.W_o
#         print(x[0,0])
#         return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_model // num_heads
        self.d_k = self.d_v

        # Define the linear layers
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)  # (d_model, d_model)
        self.W_o = nn.Linear(self.num_heads * self.d_v, self.d_model, bias=False)  # (num_heads * d_v, d_model)
        
        self.attn_pdrop = attn_pdrop

    def forward(self, x): # batch, sequence, d_model
        batch_size, seq_len, _ = x.shape
        # Split the embedding dimension to (num_heads, d_k) for q, k, and v
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # (batch, sequence, d_model) => # (batch, num_heads, sequence, d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device) # TODO make a buffer
        x = scaledDotProductAttention(q, k, v, mask, self.attn_pdrop)
        
        x = x.transpose(1, 2) # shape (batch_size,seq_len, num_heads, d_v)
        x = x.reshape(batch_size, seq_len, self.num_heads * self.d_v)
        x = self.W_o(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, residual_pdrop=0, attn_pdrop=0):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.residual_pdrop = residual_pdrop
        self.attn_pdrop = attn_pdrop

        self.rms_norm_1 = RMSNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.rms_norm_2 = RMSNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + torch.nn.functional.dropout(self.multi_head_attention(self.rms_norm_1(x)), self.residual_pdrop)
        x = x + torch.nn.functional.dropout(self.feed_forward(self.rms_norm_2(x)), self.residual_pdrop)

        # x = self.rms_norm_1(x + torch.nn.functional.dropout(self.multi_head_attention(x), self.residual_pdrop))
        # x = self.rms_norm_2(x + torch.nn.functional.dropout(self.feed_forward(x), self.residual_pdrop))

        # x = x + self.multi_head_attention(x)
        # x = x + self.feed_forward(x)

        # x = x + torch.nn.functional.dropout(self.multi_head_attention(self.rms_norm_1(x)) + self.feed_forward(self.rms_norm_2(x)), self.residual_pdrop)

        return x