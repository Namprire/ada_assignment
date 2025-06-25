import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: shape (batch_size, sequence_length, d_k)
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5  # shape: (B, T, T)
    
    # Step 2: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)  # shape: (B, T, T)
    
    # Step 3: Multiply by values
    output = torch.matmul(attention_weights, V)  # shape: (B, T, d_k)
    
    return output, attention_weights


# Dummy inputs
batch_size = 1
sequence_length = 5
d_model = 4  # dimensionality

Q = torch.randn(batch_size, sequence_length, d_model)
K = torch.randn(batch_size, sequence_length, d_model)
V = torch.randn(batch_size, sequence_length, d_model)

# Apply attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Output:", output)
print("Attention Weights:", attention_weights)
