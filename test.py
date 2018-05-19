import miniconv
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import numpy as np

np.random.seed(0)


def test_conv1():
    rand = np.random.rand(6, 6, 1)

    ker = np.array([[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]])

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs[:, :, 0] - mine.squeeze()
    ok = np.abs(diff).max() < 1e-7
    if not ok:
        print('test_conv1()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :, 0])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv2():
    rand = np.random.rand(6, 6, 1)

    ker = np.array([[[0], [0], [0]], [[0], [0], [1]], [[0], [0], [0]]])

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs[:, :, 0] - mine.squeeze()
    ok = np.abs(diff).max() < 1e-7
    if not ok:
        print('test_conv2()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :, 0])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv3():
    rand = np.random.rand(6, 6, 1)

    ker = np.random.rand(3, 5, 1)

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs.squeeze() - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
        print('test_conv3()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :, 0])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv4():
    rand = np.random.rand(6, 6, 1)

    ker1 = np.random.rand(3, 3, 1)
    ker2 = np.random.rand(3, 3, 1)

    b = miniconv.Conv(2)
    b.set_filters([ker1, ker2])

    mine = b.forward(rand)
    theirs = np.dstack([
        convolve(rand, ker1, mode='constant'),
        convolve(rand, ker2, mode='constant')
    ])

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
        print('test_conv4()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :, 0])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv3d1():
    rand = np.random.rand(6, 6, 3)

    ker = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0],
                                                        [0, 0, 0]]])

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
        print('test_conv3d1()')
        print(mine.shape)
        print(theirs.shape)
        print('rand:')
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv3d2():
    rand = np.random.rand(6, 6, 3)

    ker = np.array([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                    [[-1, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0],
                                                         [0, 0, -1]]])

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
        print('test_conv3d2()')
        print(mine.shape)
        print(theirs.shape)
        print('rand:')
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv3d3():
    rand = np.random.rand(6, 6, 3)

    ker = np.random.rand(3, 3, 3)

    b = miniconv.Conv(2)
    b.set_filters([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-5
    if not ok:
        print('test_conv3d3()')
        print(mine.shape)
        print(theirs.shape)
        print('rand:')
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv3d4():
    rand = np.random.rand(6, 6, 3)

    ker1 = np.random.rand(3, 3, 3)
    ker2 = np.random.rand(3, 3, 3)

    b = miniconv.Conv(2)
    b.set_filters([ker1, ker2])

    mine = b.forward(rand)
    theirs = np.dstack([
        convolve(rand[:, :, 0], ker1[:, :, 0], mode='constant') +
        convolve(rand[:, :, 1], ker1[:, :, 1], mode='constant') +
        convolve(rand[:, :, 2], ker1[:, :, 2], mode='constant'),
        convolve(rand[:, :, 0], ker2[:, :, 0], mode='constant') + convolve(
            rand[:, :, 1], ker2[:, :, 1], mode='constant') + convolve(
                rand[:, :, 2], ker2[:, :, 2], mode='constant')
    ])

    diff = theirs - mine
    ok = np.abs(diff).max() < 1e-5
    if not ok:
        print('test_conv3d4()')
        print(mine.shape)
        print(theirs.shape)
        print('rand:')
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        #print(theirs[:,:])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def deconv(ximg, w, dout):
    dw = np.zeros(w.shape)
    padded = np.pad(
        ximg, ((w.shape[0] // 2, w.shape[0] // 2),
               (w.shape[1] // 2, w.shape[1] // 2)),
        'constant',
        constant_values=0)
    pdx = np.zeros(padded.shape)
    for y in range(ximg.shape[0]):
        for x in range(ximg.shape[1]):
            img_y = y
            img_x = x
            for wy in range(w.shape[0]):
                for wx in range(w.shape[1]):
                    dw[wy, wx] += padded[img_y + wy, img_x + wx] * dout[y, x]
                    pdx[img_y + wy, img_x + wx] += w[wy, wx] * dout[y, x]
    dx = pdx[1:-1, 1:-1]
    return dw, dx


def test_conv_back1():
    rand = np.random.rand(6, 6, 1) * 10
    eps = np.random.rand(6, 6, 1) / 1e-5

    ker = np.array([[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]])

    b = miniconv.Conv(1)
    b.set_filters([ker])

    res2 = b.forward(rand + eps)
    res1 = b.forward(rand - eps)

    dout = np.random.rand(6, 6)
    theirs = dout * ((res2 - res1) / (2 * eps)).squeeze()
    b.forward(rand)
    mine = b.backward(dout.reshape(6, 6, 1)).squeeze()
    diff = mine - theirs
    ok = np.abs(diff).max() < 1e-7
    if not ok:
        print('test_conv_back1()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv_back2():
    rand = np.random.rand(6, 6, 1)
    dout = np.random.rand(6, 6)

    ker = np.array([[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]])

    b = miniconv.Conv(1)
    b.set_filters([ker])

    for i in range(6):
        for j in range(6):
            eps = np.zeros((6, 6, 1), dtype=float)
            eps[i, j, 0] = 1
            res1 = b.forward(rand + eps)
            res2 = b.forward(rand - eps)

            theirs = dout * ((res1 - res2) / 2).squeeze()
            b.forward(rand)
            mine = b.backward(dout.reshape(6, 6, 1)).squeeze()
            diff = mine[i, j] - theirs[i, j]
            ok = np.abs(diff).max() < 1e-6
            if not ok:
                print('test_conv_back2()')
                print(mine.shape)
                print(theirs.shape)
                print('input:')
                print(rand.squeeze())
                print('theirs:')
                print(theirs[:, :])
                print('mine:')
                print(mine.squeeze())
                print('diff:')
                print(diff)


def test_conv_back3():
    rand = np.random.rand(6, 6, 1) * 10
    dout = np.random.rand(6, 6)

    ker = np.random.rand(3, 3, 1)

    b = miniconv.Conv(1)
    b.set_filters([ker])

    for i in range(6):
        for j in range(6):
            eps = np.zeros((6, 6, 1), dtype=float)
            eps[i, j, 0] = 0.01
            res1 = b.forward(rand + eps)
            res2 = b.forward(rand - eps)

            theirs = deconv(
                rand.reshape(6, 6), ker.reshape(3, 3), dout.reshape(6, 6))[1]

            b.forward(rand)
            mine = b.backward(dout.reshape(6, 6, 1)).squeeze()
            diff = mine - theirs
            ok = np.abs(diff).max() < 1e-5
            if not ok:
                print('test_conv_back3()')
                print(i, j)
                print(mine.shape)
                print(theirs.shape)
                print(rand.squeeze())
                print('theirs:')
                print(theirs[:, :])
                print('mine:')
                print(mine.squeeze())
                print('diff:')
                print(diff)


def test_conv_back4():
    rand = np.random.rand(6, 6, 1)

    ker = np.random.rand(3, 3, 1)

    b = miniconv.Conv(1)
    b.set_filters([ker])

    for i in range(6):
        for j in range(6):
            eps = np.zeros((6, 6, 1), dtype=float)
            eps[i, j, 0] = 1
            res1 = b.forward(rand + eps)
            res2 = b.forward(rand - eps)

            theirs = ((res1 - res2) / 2).squeeze()
            b.forward(rand)
            mine = b.backward(eps).squeeze()
            diff = mine - theirs
            ok = np.abs(diff).max() < 1e-5
            if not ok:
                print('test_conv_back4()')
                print(mine.shape)
                print(theirs.shape)
                print(rand.squeeze())
                print('theirs:')
                print(theirs[:, :])
                print('mine:')
                print(mine.squeeze())
                print('diff:')
                print(diff)


def test_conv_back5():
    rand = np.random.rand(6, 6, 1)

    ker = np.random.rand(3, 3, 1)

    b = miniconv.Conv(1)
    b.set_filters([ker])

    eps = np.ones((6, 6, 1), dtype=float)
    res1 = b.forward(rand + eps)
    res2 = b.forward(rand - eps)

    theirs = ((res1 - res2) / 2).squeeze()
    b.forward(rand)
    mine = b.backward(eps).squeeze()
    diff = mine - theirs
    ok = np.abs(diff).max() < 1e-5
    if not ok:
        print('test_conv_back5()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


def test_conv_back_df1():
    rand = np.random.rand(12, 6, 1)
    ker = np.random.rand(5, 3, 1)
    dout = np.random.rand(*rand.shape)

    b = miniconv.Conv(1)

    b.set_filters([ker])
    b.forward(rand)
    b.backward(dout)

    mine = b.filters_grad()[0].squeeze()
    theirs = deconv(rand.squeeze(), ker.squeeze(), dout.squeeze())[0]

    diff = mine - theirs
    ok = np.abs(diff).max() < 1e-5
    if not ok:
        print('test_conv_back_df1()')
        print(mine.shape)
        print(theirs.shape)
        print(rand.squeeze())
        print('theirs:')
        print(theirs[:, :])
        print('mine:')
        print(mine.squeeze())
        print('diff:')
        print(diff)


test_conv1()
test_conv2()
test_conv3()
test_conv4()
test_conv3d1()
test_conv3d2()
test_conv3d3()
test_conv3d4()

test_conv_back1()
test_conv_back2()
test_conv_back3()
test_conv_back4()
test_conv_back5()
test_conv_back_df1()
