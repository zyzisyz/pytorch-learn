# -*- coding: utf-8 -*-

'''
In the above examples, we had to manually implement both the forward and 
backward passes of our neural network. 
Manually implementing the backward pass is not a big deal for a small two-layer network, 
but can quickly get very hairy for large complex networks.

Thankfully, we can use automatic differentiation to automate the computation of backward passes 
in neural networks. The autograd package in PyTorch provides exactly this functionality. 
When using autograd, the forward pass of your network will define a computational graph; 
nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. 
Backpropagating through this graph then allows you to easily compute gradients.

This sounds complicated, it’s pretty simple to use in practice. 
Each Tensor represents a node in a computational graph. 
If x is a Tensor that has x.requires_grad=True then x.grad is another Tensor holding the gradient of x with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our two-layer network; now we no longer need to manually implement the backward pass through the network:
'''

# hairy 多毛的...(居然可以这么用)