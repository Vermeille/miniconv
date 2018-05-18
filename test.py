import miniconv
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import numpy as np

np.random.seed(0)


def test_conv1():
    rand = np.random.rand(6, 6, 1)

    ker = np.array([[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]])

    b = miniconv.Conv(2)
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs[:, :, 0] - mine.squeeze()
    ok = np.abs(diff).max() < 1e-7
    if not ok:
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
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs[:, :, 0] - mine.squeeze()
    ok = np.abs(diff).max() < 1e-7
    if not ok:
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

    ker = np.random.rand(3, 3, 1)

    b = miniconv.Conv(2)
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand, ker, mode='constant')

    diff = theirs[:, :, 0] - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
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
    b.set_kernels([ker1, ker2])

    mine = b.forward(rand)
    theirs = np.dstack([
            convolve(rand, ker1, mode='constant'),
            convolve(rand, ker2, mode='constant')
            ])

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
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
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
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
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
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
    b.set_kernels([ker])

    mine = b.forward(rand)
    theirs = convolve(rand[:, :, 0], ker[:, :, 0], mode='constant')
    theirs += convolve(rand[:, :, 1], ker[:, :, 1], mode='constant')
    theirs += convolve(rand[:, :, 2], ker[:, :, 2], mode='constant')

    diff = theirs - mine.squeeze()
    ok = np.abs(diff).max() < 1e-6
    if not ok:
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
    b.set_kernels([ker1, ker2])

    mine = b.forward(rand)
    theirs = np.dstack([
         convolve(rand[:, :, 0], ker1[:, :, 0], mode='constant') +
         convolve(rand[:, :, 1], ker1[:, :, 1], mode='constant') +
         convolve(rand[:, :, 2], ker1[:, :, 2], mode='constant'),

         convolve(rand[:, :, 0], ker2[:, :, 0], mode='constant') +
         convolve(rand[:, :, 1], ker2[:, :, 1], mode='constant') +
         convolve(rand[:, :, 2], ker2[:, :, 2], mode='constant')
         ])

    diff = theirs - mine
    ok = np.abs(diff).max() < 1e-5
    if not ok:
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


test_conv1()
test_conv2()
test_conv3()
test_conv4()
test_conv3d1()
test_conv3d2()
test_conv3d3()
test_conv3d4()
