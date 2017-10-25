# MNIST Digits Classification with CNN
## Get started
1. Put `data` folder in the root path of the project
2. Run `run_cnn.py` using `python2.7`
3. Get results from log file and png plots when the training is done

## `run_cnn.py` usage
- **--name \[name\] (optional)** Your name for the model, used to name log file and png plots. Default is 'model'.
- **--lr \[lr\] (optional)** Initial learning rate, default is 0.1.
- **--momentum \[momentum\] (optional)** Momentum for SGD, default is 0.
- **--wd \[weight decay\] (optional)** Weight decay for SGD, default is 0.
- **--scheduler \[scheduler\] (optional)** Learning rate scheduler, 'empty' or 'step' for EmptyScheduler or StepScheduler. Default is 'empty'.
- **--batch \[batch\] (optional)** Training batch size, default is 128.
- **--epoch \[epoch\] (optional)** Training epoch, default is 200.
- **--init \[init stdvar\] (optional)** Initial standard variance for convolutional layers, default is 1.

## Examples
`python run_cnn.py --name mymodel --lr 0.01 --epoch 200`

`python run_cnn.py --name mymodel --lr 0.01 --momentum 0.1 --scheduler step --epoch 200`