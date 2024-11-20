from safetensors import safe_open
import torch
import sys

tensors = {}
for path in sys.argv[1:]:
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k) # loads the full tensor given a key
print(tensors)