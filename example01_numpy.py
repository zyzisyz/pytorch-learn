# -*- coding: utf-8 -*-

'''
Numpy provides an n-dimensional array object, 
and many functions for manipulating these arrays. 
Numpy is a generic framework for scientific computing; 
it does not know anything about computation graphs, or deep learning, 
or gradients. 

However we can easily use numpy to fit a 
two-layer network to random data by manually implementing the forward 
and backward passes through the network using numpy operations:
'''

import numpy as np

'''
数据准备阶段
'''

# N is batch size; D_in is input dimension;
# 啥是batch D_in是输入的维度
# H is hidden dimension; D_out is output dimension.
# H是隐层数 D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
# 创建随机输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

'''
# 接下来开始反向传播算法的表演
'''

# 学习率是什么鬼
learning_rate = 1e-6
# 迭代500次
for t in range(500):
    # Forward pass: compute predicted y
    # Forward pass: 计算预测的y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    # 计算和输出loss，注意看这个loss是怎么算的
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # 根据loss反向计算w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
