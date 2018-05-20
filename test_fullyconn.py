from miniconv import FullyConn, MSE
import numpy as np
import time

np.random.seed(0)

fc = FullyConn()
mse = MSE()

x = np.random.rand(5, 6, 3)
w = np.random.rand(5, 6, 3)

fc.set_weights(w, 0)
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

x = np.random.rand(1, 1, 1)
t_x = np.random.rand(*x.shape)
b = 0#np.random.rand()
fc.set_weights(np.random.rand(*x.shape), 0)
for i in range(100):
    start = time.time()
    for j in range(100):
        x = np.random.rand(*x.shape)
        y = (x * t_x).sum() + b
        mse.set_target(y.reshape(1, 1, 1))
        out = mse.forward(fc.forward(x))
        fc.backward(mse.backward(np.array([[[1]]])))
    fc.update(0.01)
    print('res: ', np.abs(fc.weights() - t_x).squeeze(), fc.bias(), b)
