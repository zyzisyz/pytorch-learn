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
# batch size
'''
机器学习中参数更新的方法有三种：
1.  Batch Gradient Descent，
    批梯度下降，遍历全部数据集计算一次损失函数，
    进行一次参数更新，这样得到的方向能够更加准确的指向极值的方向，但是计算开销大，速度慢；

2.  Stochastic Gradient Descent，
    随机梯度下降，对每一个样本计算一次损失函数，
    进行一次参数更新，优点是速度快，缺点是方向波动大，忽东忽西，不能准确的指向极值的方向，
    有时甚至两次更新相互抵消；

3. Mini-batch Gradient Decent，
    小批梯度下降，前面两种方法的折中，把样本数据分为若干批，分批来计算损失函数和更新参数，
    这样方向比较稳定，计算开销也相对较小。Batch Size就是每一批的样本数量。


Iteration：迭代，可以理解为w和b的一次更新，就是一次Iteration。

Epoch：样本中的所有样本数据被计算一次就叫做一个Epoch。
'''

# D_in是输入的维度

# H is hidden dimension; D_out is output dimension.
# H是隐层数 
# D_out是输出维度

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

# 为了能够使得梯度下降法有较好的性能，我们需要把学习率的值设定在合适的范围内
# 学习率决定了参数移动到最优值的速度快慢。如果学习率过大，很可能会越过最优值；
# 反而如果学习率过小，优化的效率可能过低，长时间算法无法收敛。
# 所以学习率对于算法性能的表现至关重要。
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
