from miniconv import Net
import numpy as np

t_x = np.random.rand(10, 1, 1)
xs = [np.random.rand(10, 1, 1) for _ in range(1024) ]
ys = [(t_x * x).sum().reshape(1, 1, 1) for x in xs]
net = Net()
net.set_lr(0.02)
net.set_batch_size(32)
net.set_epochs(3)
net.input(10, 1, 1)
net.fc()
net.train(xs, ys)
