from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO, beautiful_dict
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
from scheduler import EmptyScheduler, StepScheduler
import matplotlib
from sys import platform
if platform == 'linux' or platform == 'linux2':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from visualize import vis_conv

if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        default='model',
        help="""\
        Model name\
        """
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help="""\
        Initial learning rate\
        """
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.0,
        help="""\
        Momentum\
        """
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=100,
        help="""\
        Batch size\
        """
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        help="""\
        Epoch number\
        """
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0,
        help="""\
        Weight decay\
        """
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='empty',
        help="""\
        Learning rate scheduler\
        """
    )
    parser.add_argument(
        '--init',
        type=float,
        default=1,
        help="""\
        Conv filter initialization\
        """
    )
    args = parser.parse_args()

    # use model name to determine log filename and plot filename
    model_name = args.name
    log_file = model_name + '.log'
    loss_plot_file = model_name + '_loss.png'
    acc_plot_file = model_name + '_acc.png'

    # read train and test data
    train_data, test_data, train_label, test_label = load_mnist_4d('data')

    # Your model defintion here
    # You should explore different model architecture
    model = Network()
    model.add(Conv2D('conv1', 1, 4, 3, 1, args.init))
    model.add(Relu('relu1'))
    model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
    model.add(Conv2D('conv2', 4, 4, 3, 1, args.init))
    model.add(Relu('relu2'))
    model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
    model.add(Reshape('flatten', (-1, 196)))
    model.add(Linear('fc3', 196, 10, 0.1))
    # write model architecture to log file
    LOG_INFO('Model architecture:\r\n' + str(model) + '\r\n', time=False, to_file=log_file)

    # use cross entropy loss function
    loss = SoftmaxCrossEntropyLoss(name='loss')

    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.

    config = {
        'learning_rate': args.lr,
        'weight_decay': args.wd,
        'momentum': args.momentum,
        'batch_size': args.batch,
        'max_epoch': args.epoch,
        'disp_freq': 0,
        'test_epoch': 1
    }
    # write configurations to log file
    LOG_INFO('Configurations:\r\n' + beautiful_dict(config) + '\r\n', time=False, to_file=log_file)

    # choose learning rate scheduler based on command line argument
    scheduler_str = args.scheduler
    if scheduler_str == 'empty':
        # EmptyScheduler does nothing to learning rate
        scheduler = EmptyScheduler(config['learning_rate'])
    elif scheduler_str == 'step':
        # StepScheduler decay learning rate by 0.1 every 10 epochs
        scheduler = StepScheduler(config['learning_rate'], step_size=50, decay=0.5, min_lr=1e-6)
    else:
        raise Exception('Scheduler named {} not found.'.format(scheduler_str))

    # loss list for plot
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # start training procedure
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch), to_file=log_file)
        config['learning_rate'] = scheduler.step(epoch)
        train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        msg = '    Training, total mean loss %.5f, total acc %.5f' % (train_loss, train_acc)
        LOG_INFO(msg, to_file=log_file)

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch), to_file=log_file)
            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            msg = '    Testing, total mean loss %.5f, total acc %.5f' % (test_loss, test_acc)
            LOG_INFO(msg, to_file=log_file)

    LOG_INFO('Best train loss: %.5f' % (min(train_loss_list)), to_file=log_file)
    LOG_INFO('Best train acc: %.5f' % (max(train_acc_list)), to_file=log_file)
    LOG_INFO('Best test loss: %.5f' % (min(test_loss_list)), to_file=log_file)
    LOG_INFO('Best test acc: %.5f' % (max(test_acc_list)), to_file=log_file)

    # plot train and test loss using matplotlib
    x = range(1, config['max_epoch'] + 1)
    plt.title('Train/Test Loss v.s. Epoch')
    plt.ylim((0, 0.8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, train_loss_list, 'r', label='Train')
    plt.plot(x, test_loss_list, 'b', label='Test')
    plt.legend(loc='upper right')
    plt.savefig(loss_plot_file)
    plt.clf()
    plt.title('Train/Test Accuracy v.s. Epoch')
    plt.ylim((0.5, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(x, train_acc_list, 'r', label='Train')
    plt.plot(x, test_acc_list, 'b', label='Test')
    plt.legend(loc='lower right')
    plt.savefig(acc_plot_file)
    plt.clf()

    # plot conv1 feature maps
    if not (platform == 'linux' or platform == 'linux2'):
        vis_conv(model, train_data, train_label, 4, 'conv1')
        vis_conv(model, train_data, train_label, 4, 'conv2')
