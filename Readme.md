# DotML
Machine Learning practice implementations. 

## Folders
| Name | Purpose |
|------|---------|
| Data | Raw data used to create datasets |
| DotML | Core library with ML implementations |
| DotML.Sandbox | Blazor Server app for experimenting with machine learning scenarios |
| DotML.Test | Test library for validating parts of the code non-experimentally |
| DotML.Utils | Utility projects, often to create training data sets |

## Current Progress
[x] Classical Fully Connected Networks
[1/2] Convolutional Networks - Currently stuck on failed training for larger networks like AlexNet and MobileNet
    - NaNs occasionally appear
    - Slow convergence (with tanh)
    - ReLU never seems to have any convergence at all
    - Training is SLOW, minimizing how often changes can be made
[ ] Residual Networks
[ ] UNet
[ ] Stable Diffusion
[ ] Transformers 