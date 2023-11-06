# -*- coding: utf-8 -*-
import torch
import math


dtype = torch.float
device = torch.device("cpu")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
# Apart from that, it is imperative to check whether or not the program runs natively on the Apple M1 processor. If this is the case, under ‘Kind’ on the Activity Monitor you should see the ‘Apple’ option. If you see Intel, something went wrong during the installation.

# Addressing a few common issues:
# 1 . Numpy not recognized
# Actually, this issue occurs due to Miniforge installing a non conda-forge numpy version. Probably you would see something like this:

# This issue can be easily fixed by installing the openblas package:
# $ conda install -c conda-forge openblas
# 2. Intel usage instead of Apple:
# Some users may still observe in the Activity Monitor that Intel is used instead of Apple. One possible explanation for this is that conda did not correctly identified the arm64 architecture (which is done automatically). As a consequence, conda does install the arm64-based version of the libraries.
# In that case, the architecture should be specified explicitly when creating a new conda environment. To take care of this issue, when creating new conda environment in Minforge execute this instead:
# $ CONDA_SUBDIR=osx-arm64 conda create --name pytorch_m1 python=3.8
# Closing Remarks
# Pytorch is an integral part of the data science ecosystem, and that’s why I believe a comprehensive guide for installing that framework natively on the new series of Apple chip computers is absolutely necessary.
# On the other hand, Pytorch has still a long way to go. The next milestone would probably be a library or a plugin that would allow Pytorch to utilize the GPU.
