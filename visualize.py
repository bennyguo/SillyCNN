from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from solve_net import data_iterator

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.axis('off')
    plt.imshow(data, cmap='gray')
    plt.show()

def vis_conv(model, inputs, labels, batch_size, vis_layer):
    count = 0
    for input, label in data_iterator(inputs, labels, batch_size):
        layer_names = [layer.name for layer in model.layer_list]
        layer_idx = layer_names.index(vis_layer)
        # forward net
        for i in range(layer_idx + 1):
            input = model.layer_list[i].forward(input)
        features = input
        features = features.reshape(features.shape[0] * features.shape[1], features.shape[2], features.shape[3])
        vis_square(features)
        count += 1
        if(count == 5):
            break