Miniconv
========

Miniconv is a minimalistic, CPU-based, but intended to be fast, neural networks
C++ library, also coming with a python frontend.

It's _intended_ to be fast but TensorFlow, even on CPU, is significantly
faster.

Minimalistic also means that it's less than feature complete. It just
implements basic and common needs without a lot of tuning options.

Said otherwise, if your neural net is the main element of your pipeline, don't
use Miniconv. If it's _not_, and you'd want to train and implement a small
network for a little task, scikit-learn doesn't fit your needs, and TF is
consider too heavy, that's what Miniconv is for.

Volume
======

A `Volume` is a 3D array used internally by Miniconv. All numpy arrays are
changed into `Volume` when getting into Miniconv. That means that _all numpy
arrays passed into Miniconv must be 3D_.

Net
===

Overview
--------

The preferred way to use Miniconv is through the `Net` object. This object
allows you to specify various hyper parameters and the layers composing your
network.

```python3
net = Net()           # Create an empty network
net.set_lr(0.01)      # set the learning rate

net.input(32, 32, 3)  # input layer, of shape (32, 32, 3)
net.conv(3, 3, 64)    # conv layer of 64 kernels of size 3x3
net.relu()            # relu activation
net.fc(32)            # fully connected layer with 32 output units
net.tanh()            # tanh activation
net.fc(1)             # fully connected layer with 1 output unit

net.train(X, y)       # X and y are lists of 3D np arrays

print(net.predict(example))
```

Params
------

* `set_batch_size(int)`: size of batches
* `set_epochs(int)`: number of training epochs
* `set_grad_max(float)`: gradient clipping maximum value
* `set_l2(float)`: L2 regularizer parameter
* `set_lr(float)`: learning rate
* `set_lr_decay(float)`: `lr *= decay` happens at each epoch
* `set_optimizer(Optimizer)`: optimizer to use. `Optimizer` is an enum with
  several values:

  * `Optimizer.sgd`: standard SGD
  * `Optimizer.sgdm`: SGD with momentum
  * `Optimizer.powersign`

Layers
------

* `input(int w, int h, int c)`: the input layer MUST be your first layer. It
  takes data of width `w`, height`h` and `c` channels.
* `conv(int w, int h, int nb)`: a conv layer with `nb` filter of width `w` and
  height `h`. Convs have biases and are "same" convolution (keeping the shape)
* `maxpool()`: a 2x2, "valid", maxpool layer. It ignores the last row / column
  if their size is odd.
* `batch_norm()`: This is _not_ a proper batch norm, though it keeps a moving
  average of the mean and standard deviation of the data passing trough it,
  rescales it with to attempt a mean of 0 and std of 1. Then it's rescaled
  again with std `gamma` and mean `beta` which are both learnt parameters.
* `relu()`
* `tanh()`
* `verbose(str name)`: A debugging layer. It saves data passing through it in
  forward and backward passes during a batch, and displays some statistics
  during parameters update. Don't hesitate to place several `verbose` layers at
  various points of your net to inspect it. Name them how it pleases you.

Methods
-------

* `train(Xs, ys)`: start training. `Xs` and `ys` have to be lists of np 3D
  arrays.
* `predict(x)`: runs `x` through the network and returns the output.

Layers
======

The layers can also be used outside of a `Net` so that you can have a bit more
control over them. Though doing so is discouraged since it would mean
converting np arrays to `Volume`s back and forth.

