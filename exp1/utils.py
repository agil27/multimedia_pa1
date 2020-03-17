import numpy as np


def zigzag(n):
    '''
    returns a zigzag-ordered marking nxn matrix.
    e.g. for n=4, returns
    0  1  5  6
    2  4  7 12
    3  8 11 13
    9 10 14 15
    '''
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # find the diagonal
            s = i + j
            if i + j < n:
                # upper half of the matrix
                z[i, j] = s * (s + 1) / 2
                z[i, j] += i if s % 2 == 1 else s - i
            else:
                t = 2 * n - 2 - s
                z[i, j] = t * (t + 1) / 2
                z[i, j] += n - 1 - i if t % 2 == 1 else t + 1 + i - n
                z[i, j] = n ** 2 - 1 - z[i, j]
    return z


def compress(x, comp_coef):
    '''compresses the dct result by zigzag path'''
    n = x.shape[0]
    preserve = comp_coef * (n ** 2)
    ans = (zigzag(n) < preserve) * x
    return ans


def dct_matrix(n):
    c = np.ones((n, n))
    c[0, :] = np.sqrt(1 / n)
    for i in range(1, n):
        for j in range(n):
            c[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n)) * np.sqrt(2 / n)
    return c


def blockproc(x, block, func, *args, **kwargs):
    m = x.shape[0]
    n = x.shape[1]
    stride_x = block[0]
    stride_y = block[1]
    ans = np.zeros_like(x)
    for i in range(0, m, stride_x):
        for j in range(0, n, stride_y):
            view = x[i: i + stride_x, j: j + stride_y]
            ans[i: i + stride_x, j: j + stride_y] = func(view, *args, **kwargs)
    return ans


def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def dct(x, inverse=False, c=None):
    dim = len(x.shape)
    n = x.shape[0]
    if c is None:
        c = dct_matrix(n)

    if dim == 1:
        if inverse is False:
            return np.dot(c, x)
        else:
            return np.dot(np.transpose(c), x)
    else:
        if inverse is False:
            return np.dot(np.dot(c, x), np.transpose(c))
        else:
            return np.dot(np.dot(np.transpose(c), x), c)


def idct(x, c=None):
    return dct(x, inverse=True, c=c)
