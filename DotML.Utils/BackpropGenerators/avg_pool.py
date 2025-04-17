import torch
import torch.nn as nn
import json

# Step 1: Make layer
layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

# Step 2: Create random input tensor
input = torch.randn(1, 1, 5, 5, requires_grad=True) # 1 batch, 1 channel, 5x5 feature

# Step 3: Forward pass
output = layer(input)

# Step 4: Define the output gradient
dY = torch.randn_like(output)

# Step 5: Backpropagate
output.backward(dY)

# Step 6: Convert tensors to lists
input_list = input.tolist()
output_list = output.tolist()
#W_list = layer.weight.tolist()
#B_list = layer.bias.tolist()
dY_list = dY.tolist()
dX_list = input.grad.tolist()
#dW_list = layer.weight.grad.tolist()
#dB_list = layer.bias.grad.tolist()

output = {
    'X': input_list,
    #'W': W_list,
    #'B': B_list,
    'Y': output_list,
    'dY': dY_list,
    'dX': dX_list,
    #'dW': dW_list,
    #'db': dB_list
}

# Step 7: Save
filename = 'avg_pool.tensors.json'
with open(filename, 'w') as file:
    json.dump(output, file, indent=4)
print(f"Tensors and gradients saved to '{filename}'.")