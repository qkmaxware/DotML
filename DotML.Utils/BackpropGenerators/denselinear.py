import torch
import torch.nn as nn
import json

# Step 1: Make layer
in_feats = 5
out_feats = 3
layer = nn.Linear(in_feats, out_feats, True)

# Step 2: Create random input tensor
input = torch.randn(1, 1, 1, in_feats, requires_grad=True) # 1 batch, 1 channel, 1x5 feature

# Step 3: Forward pass
output = layer(input)

# Step 4: Define the output gradient
dY = torch.randn_like(output)

# Step 5: Backpropagate
output.backward(dY)

# Step 6: Convert tensors to lists
input_list = input.view(in_feats, 1).detach().tolist()
output_list = output.view(out_feats, 1).detach().tolist()
W_list = layer.weight.detach().tolist()
B_list = layer.bias.detach().tolist()
dY_list = dY.view(out_feats, 1).detach().tolist()
dX_list = input.grad.view(in_feats, 1).detach().tolist()
dW_list = layer.weight.grad.tolist()
dB_list = layer.bias.grad.tolist()

output = {
    'X': input_list,
    'W': W_list,
    'B': B_list,
    'Y': output_list,
    'dY': dY_list,
    'dX': dX_list,
    'dW': dW_list,
    'db': dB_list
}

# Step 7: Save
filename = 'denselinear.tensors.json'
with open(filename, 'w') as file:
    json.dump(output, file, indent=4)
print(f"Tensors and gradients saved to '{filename}'.")