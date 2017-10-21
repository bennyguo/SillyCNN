import numpy as np
from functions import *

input = np.random.rand(128, 3, 14, 14)
grad_output = np.random.rand(128, 10, 14, 14)
W = np.random.rand(10, 3, 3, 3)
b = np.random.rand(10)
grad_input1, _, _ = conv2d_backward(input, grad_output, W, b, 3, 1)
grad_input2 = conv2d_backward_test(input, grad_output, W, b, 3, 1)
print(np.allclose(grad_input1, grad_input2))