from block import *
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, context_length: int, rope_theta: float):
        super().__init__()
        self.embedding = Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.atten_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, d_model,d_model, d_model, d_model, context_length, rope_theta) for _ in range(num_layers)])
        self.rms = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
    
    def forward(self,  in_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_indices)
        for layer in self.atten_layers:
            x = layer(x)
        return self.output_linear(self.rms(x))