import torch
import torch.nn as nn
import math
from einops import einsum, reduce, rearrange
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.dtype = dtype
        self.device = device
        sigma_2 = 2 / (in_features + out_features)
        sigma = math.sqrt(sigma_2)
        nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.device:
            x = x.to(self.device)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings,embedding_dim))
        nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)
        self.device = device
        self.dtype = dtype
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5,device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.eps = eps

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_type = x.dtype
        x = x.to(torch.float32)
        ms = reduce(x**2, "... d -> ... 1", "mean")
        ms_inv = torch.rsqrt(ms + self.eps)
        result = x * ms_inv * self.g
        return result.to(in_type)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int = None):
        super().__init__()
        self.d_ff = d_ff
        if d_ff is None:
            self.d_ff = (d_model * 8 ) // 3
            self.d_ff = ((self.d_ff + 63) // 64) * 64
        self.d_model = d_model
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff,d_model)
        self.linear3 = Linear(d_model, d_ff)
        self.silu = SiLU()
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        theta = 1.0 / (self.theta ** (torch.arange(0, d_k, 2) / d_k)) #(d_k/2)
        m = torch.arange(max_seq_len,dtype=torch.float32)
        freq = torch.outer(m, theta) # (max_seq_len, d_k/2)
        freq_repetead = torch.repeat_interleave(freq, 2, dim=-1) # (max_seq_len, d_k)
        self.register_buffer("cos_cached", freq_repetead.cos(), persistent=False)
        self.register_buffer("sin_cached", freq_repetead.sin(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # (max_sqe_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_rotated = rearrange(
    torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1),
    "... d_half two -> ... (d_half two)",  
    two=2                                  
)
        return x * cos + x_rotated * sin

class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x:torch.Tensor, dim:int = -1) -> torch.Tensor:
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max
        x_exp = torch.exp(x_stable)
        x_sum = x_exp.sum(dim=dim, keepdim=True)
        return x_exp / x_sum
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = SoftMax()
    def forward(self,Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
        at = einsum(Q, K, "... q_len d_model, ... k_len d_model -> ... q_len k_len")*torch.rsqrt(torch.tensor(Q.shape[-1]))
        if mask is not None:
            at = at.masked_fill(~mask, float('-inf'))
        scores = self.sm(at,-1)
        return einsum(scores, V, "... q_len k_len, ... k_len d_model -> ... q_len d_model")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_in:int, d_q:int, d_k:int, d_v: int, 
                 use_rope: bool = False, max_seq_len: int = None, theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k // num_heads  # per-head dimension
        self.linear_q = Linear(d_in, d_q )
        self.linear_k = Linear(d_in, d_k )
        self.linear_v = Linear(d_in, d_v )
        self.linear_o = Linear(d_v , d_model)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # x: [... seq_len, d_in]
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 
        q = rearrange(q, '... seq_len (h d) -> ... h seq_len d', h = self.num_heads)
        k = rearrange(k, '... seq_len (h d) -> ... h seq_len d', h = self.num_heads)
        v = rearrange(v, '... seq_len (h d) -> ... h seq_len d', h = self.num_heads)
            
        if self.use_rope:
            if token_positions is None:
                seq_len = q.shape[-2]
                token_positions = torch.arange(seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        mask = torch.tril(torch.ones(q_len, k_len), diagonal=0).bool()
        atten_out = self.ScaledDotProductAttention(q, k, v, mask = mask)
        return self.linear_o(rearrange(atten_out, '... h s d -> ... s (h d)'))

# 输入x, 输出: x + MHA(RMS(x))
class MHALayer(nn.Module):
    def __init__(self, d_model:int, num_heads: int, d_in: int, d_q: int, d_k: int, d_v: int, max_seq_len: int,theta:float):
        super().__init__()
        self.rms = RMSNorm(d_model=d_model)
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, d_in = d_in, d_q = d_q, d_k = d_k, d_v = d_v,use_rope=True,max_seq_len=max_seq_len,theta=theta)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x + self.mhsa(self.rms(x))

class FFNLayer(nn.Module):
    def __init__(self,d_model:int, d_ff:int):
        super().__init__()
        self.swiglu = SwiGLU(d_model = d_model, d_ff = d_ff)
        self.rms = RMSNorm(d_model=d_model)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x + self.swiglu(self.rms(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, d_in: int, d_q: int, d_k: int, d_v: int, max_seq_len: int,theta:float):
        super().__init__()
        self.mha_layer = MHALayer(d_model=d_model, num_heads=num_heads, d_in=d_in, d_q = d_q, d_k = d_k, d_v = d_v, max_seq_len=max_seq_len, theta=theta)
        self.ffn_layer = FFNLayer(d_model=d_model,d_ff=d_ff)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.ffn_layer(self.mha_layer(x))
