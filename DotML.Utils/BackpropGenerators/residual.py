import torch
import torch.nn as nn
import json
 
# --------------------------
# Config
# --------------------------
in_feats = 5
hidden_feats = 4
out_feats = 3
 
# --------------------------
# Residual block definition
# --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features, bias=True)
        )
        # Skip path (projection if dimensions differ)
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=True)
        else:
            self.skip = nn.Identity()
   
    def forward(self, x):
        return self.main_path(x) + self.skip(x)
 
# --------------------------
# Create block
# --------------------------
res_block = ResidualBlock(in_feats, hidden_feats, out_feats)
 
# --------------------------
# Input
# --------------------------
X = torch.randn(1, in_feats, requires_grad=True)  # batch=1
Y = res_block(X)
 
# --------------------------
# Random gradient for backprop
# --------------------------
dY = torch.randn_like(Y)
Y.backward(dY)
 
# --------------------------
# Helper to flatten and convert tensor to list
# --------------------------
def tensor_to_list(t):
    return t.detach().view(-1).tolist() if t is not None else None
 
# --------------------------
# Collect all tensors
# --------------------------
output = {
    'X': tensor_to_list(X),
    'Y': tensor_to_list(Y),
    'dY': tensor_to_list(dY),
   
    # Main path weights & grads
    'main_W1': tensor_to_list(res_block.main_path[0].weight),
    'main_b1': tensor_to_list(res_block.main_path[0].bias),
    'main_dW1': tensor_to_list(res_block.main_path[0].weight.grad),
    'main_db1': tensor_to_list(res_block.main_path[0].bias.grad),
 
    'main_W2': tensor_to_list(res_block.main_path[2].weight),
    'main_b2': tensor_to_list(res_block.main_path[2].bias),
    'main_dW2': tensor_to_list(res_block.main_path[2].weight.grad),
    'main_db2': tensor_to_list(res_block.main_path[2].bias.grad),
   
    # Skip path
    'skip_W': tensor_to_list(res_block.skip.weight) if hasattr(res_block.skip, 'weight') else None,
    'skip_b': tensor_to_list(res_block.skip.bias) if hasattr(res_block.skip, 'bias') else None,
    'skip_dW': tensor_to_list(res_block.skip.weight.grad) if hasattr(res_block.skip, 'weight') else None,
    'skip_db': tensor_to_list(res_block.skip.bias.grad) if hasattr(res_block.skip, 'bias') else None,
   
    'dX': tensor_to_list(X.grad)
}
 
# --------------------------
# Save to JSON
# --------------------------
filename = 'residual_block.tensors.json'
with open(filename, 'w') as f:
    json.dump(output, f)
print(f"Tensors and gradients saved to '{filename}'.")