import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5  # (B, T, T)
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V
    return output, attention_weights

B, T, d = 1, 5, 4
Q = torch.randn(B, T, d)
K = torch.randn(B, T, d)
V = torch.randn(B, T, d)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Output:", output)
print("Attention Weights:", attention_weights)