from __future__ import division
import numpy as np
from im2col_cython import col2im_cython, im2col_cython


def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    batch, _, hin, win = input.shape
    cout, cin, _, _ = W.shape
    # input_mat = im2col(input, kernel_size, kernel_size, pad)
    input_mat = im2col_cython(input, kernel_size, kernel_size, pad, 1)
    W_mat = W.reshape(cout, -1)
    hout = hin + 2 * pad - kernel_size + 1
    wout = win + 2 * pad - kernel_size + 1
    output = W_mat.dot(input_mat) + b.reshape(-1, 1)
    return output.reshape(cout, hout, wout, batch).transpose(3, 0, 1, 2), input_mat


def conv2d_backward(input, grad_output, W, b, kernel_size, pad, cache):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
    batch, cin, hin, win = input.shape
    _, cout, hout, wout = grad_output.shape
    input_mat = cache
    # grad_output_mat = im2col(grad_output, kernel_size, kernel_size, kernel_size - 1)
    grad_output_mat = im2col_cython(grad_output, kernel_size, kernel_size, kernel_size - 1, 1)
    W_mat = np.rot90(W, 2, (2, 3)).transpose(1, 0, 2, 3).reshape(cin, -1)
    grad_input = W_mat.dot(grad_output_mat).reshape(cin, hin + 2 * pad, win + 2 * pad, batch)[:, pad:hin + pad, pad:win + pad, :].transpose(3, 0, 1, 2)
    grad_W = grad_output.transpose(1, 0, 2, 3).reshape(cout, -1).dot(input_mat.T).reshape(W.shape)
    grad_b = np.sum(grad_output, axis=(0, 2, 3))
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    batch, cin, hin, win = input.shape
    hout, wout = (hin + 2 * pad) // kernel_size, (win + 2 * pad) // kernel_size
    # input_mat = im2col(input, kernel_size, kernel_size, pad, kernel_size)
    input_mat = im2col_cython(input, kernel_size, kernel_size, pad, kernel_size)
    input_mat = input_mat.reshape(cin, kernel_size * kernel_size, hout, wout, batch).transpose(1, 4, 0, 2, 3)
    return np.mean(input_mat, axis=0)


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    batch, cin, hin, win = input.shape
    _, _, hout, wout = grad_output.shape
    grad_output_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(cin, -1)
    grad_input_mat = np.repeat(grad_output_reshaped, kernel_size * kernel_size, axis=0) / float(kernel_size * kernel_size)
    # grad_input = col2im(grad_input_mat, input.shape, kernel_size, kernel_size, pad, kernel_size)[:, :, pad:pad + hin, pad:pad + win]
    grad_input = col2im_cython(grad_input_mat, batch, cin, hin, win, kernel_size, kernel_size, pad, kernel_size)[:, :, pad:pad + hin, pad:pad + win]
    return grad_input

