import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

# Configuration
d_model = 4      # Embedding/input size
d_k = 4          # Output size per head
heads = 1        # Number of attention heads
batch_size = 1
seq_len = 3

# Create projection layers
q_proj = nn.Linear(d_model, d_k, bias=True)
k_proj = nn.Linear(d_model, d_k, bias=True)
v_proj = nn.Linear(d_model, d_k, bias=True)
output_proj = nn.Linear(d_k * heads, d_model, bias=True)

# Create random input: [batch, seq_len, d_model]
X = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

# Forward pass: Project input to Q, K, V
Q_base = q_proj(X)  # [batch, seq_len, d_k]
K_base = k_proj(X)  # [batch, seq_len, d_k]
V_base = v_proj(X)  # [batch, seq_len, d_k]

# Replicate projections per head: [batch, seq_len, d_k] -> [batch, heads, seq_len, d_k]
Q = Q_base.unsqueeze(1).expand(-1, heads, -1, -1)
K = K_base.unsqueeze(1).expand(-1, heads, -1, -1)
V = V_base.unsqueeze(1).expand(-1, heads, -1, -1)

# Compute scaled dot-product attention: Q @ K^T / sqrt(d_k)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, heads, seq_len, seq_len]

# Apply softmax
attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]

# Apply attention to values
attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, d_k]

# Concatenate heads: [batch, heads, seq_len, d_k] -> [batch, seq_len, heads*d_k]
attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
attn_output = attn_output.view(batch_size, seq_len, -1)

# Final output projection back to d_model
Y = output_proj(attn_output)  # [batch, seq_len, d_model]

# Define output gradient
dY = torch.randn_like(Y)

# Backward pass
Y.backward(dY)

# Helper to convert tensors to lists
def tensor_to_list(t):
    return t.detach().cpu().tolist()

print(X.shape)
print(Q_base.shape)
print(K_base.shape)
print(V_base.shape)
print(Q.shape)
print(K.shape)
print(V.shape)
print(scores.shape)
print(attn_weights.shape)
print(attn_output.shape)
print(Y.shape)
print(dY.shape)
print(X.grad.shape)

print(q_proj.weight.shape)
print(q_proj.bias.shape)
print(k_proj.weight.shape)
print(k_proj.bias.shape)
print(v_proj.weight.shape)
print(v_proj.bias.shape)

print(q_proj.weight.grad.shape)
print(q_proj.bias.grad.shape)
print(k_proj.weight.grad.shape)
print(k_proj.bias.grad.shape)
print(v_proj.weight.grad.shape)
print(v_proj.bias.grad.shape)
print(output_proj.weight.grad.shape)
print(output_proj.bias.grad.shape)

output = {
    'X': tensor_to_list(X),
    'Q_base': tensor_to_list(Q_base),
    'K_base': tensor_to_list(K_base),
    'V_base': tensor_to_list(V_base),
    'Q': tensor_to_list(Q),
    'K': tensor_to_list(K),
    'V': tensor_to_list(V),
    'scores': tensor_to_list(scores),
    'attn_weights': tensor_to_list(attn_weights),
    'attn_output_per_head': tensor_to_list(attn_output),
    'Y': tensor_to_list(Y),
    'dY': tensor_to_list(dY),
    'dX': tensor_to_list(X.grad),
    'Q_proj_W': tensor_to_list(q_proj.weight),
    'Q_proj_b': tensor_to_list(q_proj.bias),
    'K_proj_W': tensor_to_list(k_proj.weight),
    'K_proj_b': tensor_to_list(k_proj.bias),
    'V_proj_W': tensor_to_list(v_proj.weight),
    'V_proj_b': tensor_to_list(v_proj.bias),
    'dQ_proj_W': tensor_to_list(q_proj.weight.grad),
    'dQ_proj_b': tensor_to_list(q_proj.bias.grad),
    'dK_proj_W': tensor_to_list(k_proj.weight.grad),
    'dK_proj_b': tensor_to_list(k_proj.bias.grad),
    'dV_proj_W': tensor_to_list(v_proj.weight.grad),
    'dV_proj_b': tensor_to_list(v_proj.bias.grad),
    'Output_proj_W': tensor_to_list(output_proj.weight),
    'Output_proj_b': tensor_to_list(output_proj.bias),
    'dOutput_proj_W': tensor_to_list(output_proj.weight.grad),
    'dOutput_proj_b': tensor_to_list(output_proj.bias.grad),
}

# Save to JSON
filename = f'selfattention.{d_model}-{d_k}-{heads}.tensors.json'
with open(filename, 'w') as f:
    json.dump(output, f)
print(f"Tensors and gradients saved to '{filename}'.")