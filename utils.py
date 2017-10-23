from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime


def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


# print msg to console
# print time if [time] is true
# log to file [to_file] if [to_file] is not None
def LOG_INFO(msg, time=True, to_file=None):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    full_msg = ((display_now + ' ') if time else '') + msg
    print(full_msg)
    if to_file is not None:
        with open(to_file, 'a') as f:
            f.write(full_msg + '\r\n')


# return pretty string of a dict for print use
def beautiful_dict(d, pre=''):
    if not isinstance(d, dict):
        raise Exception('Expected a dict, got {} instead',format(type(d)))
    result = ''
    for key, value in d.items():
        if isinstance(value, dict):
            result += pre + '{}:\r\n'.format(key)
            result += beautiful_dict(value, pre+'\t')
        else:
            result += pre + '{}: {}\r\n'.format(key, value)
    return result
