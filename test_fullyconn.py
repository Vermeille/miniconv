from miniconv import FullyConn, MSE
import numpy as np
import time

np.random.seed(0)

fc = FullyConn()
mse = MSE()

x = np.random.rand(5, 6, 3)
w = np.random.rand(5, 6, 3)

fc.set_weights(w, np.array([[[0]]]))
mine = fc.forward(x).squeeze()
theirs = (x * w).sum()
diff = mine - theirs
ok = np.abs(diff) < 1e-5
if not ok:
    print('fail forward')
    print(x)
    print('theirs:')
    print(theirs)
    print('mine:')
    print(mine)
    print('diff:')
    print(diff)

pgrad = np.random.rand(1, 1, 1)
theirs = w * pgrad
fc.forward(x)
mine = fc.backward(pgrad)
diff = theirs - mine
ok = np.abs(diff).max() < 1e-5
if not ok:
    print('fail forward')
    print(x)
    print('theirs:')
    print(theirs)
    print('mine:')
    print(mine)
    print('diff:')
    print(diff)
mine = fc.grads()
theirs = x * pgrad
diff = mine - theirs
ok = np.abs(diff).max() < 1e-6
if not ok:
    print('fail forward')
    print(x)
    print('theirs:')
    print(theirs)
    print('mine:')
    print(mine)
    print('diff:')
    print(diff)

x = np.random.rand(4, 1, 1)
t_x = np.random.rand(*x.shape)
b = np.random.rand()
fc.set_weights(np.random.rand(*x.shape), np.array([[[0]]]))
for i in range(50):
    outs = []
    for j in range(32):
        x = np.random.rand(*x.shape)
        y = (x * t_x).sum() + b
        mse.set_target(y.reshape(1, 1, 1))
        out = mse.forward(fc.forward(x))
        fc.backward(mse.backward(np.array([[[1]]])))
        outs.append(out)
    print(np.abs(t_x - fc.weights()).squeeze(), b - fc.bias())
    fc.update(0.01)
    #print('loss: ', np.mean(outs))
print(np.abs(t_x - fc.weights()).squeeze(), b - fc.bias())
