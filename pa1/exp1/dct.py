import cv2
import numpy as np
import time

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


def perform_dct_1d(img, dct_func, idct_func, comp_coef):
    start = time.process_time()
    n = img.shape[0]
    dct1drow = np.zeros_like(img)
    dct1d = np.zeros_like(img)
    c = dct_matrix(n)
    for i in range(n):
        dct1drow[i, :] = dct_func(img[i, :], c=c).squeeze()
    for i in range(n):
        dct1d[:, i] = dct_func(dct1drow[:, i], c=c).squeeze()

    idct1d = np.zeros_like(img)
    for i in range(n):
        idct1d[i, :] = idct_func(dct1d[i, :], c=c).squeeze()
    for i in range(n):
        idct1d[:, i] = idct_func(idct1d[:, i], c=c).squeeze()
    end = time.process_time()
    print('dct1d_time: %.2f' % (end - start))

    cv2.imwrite('dct1drow.bmp', dct1drow)
    cv2.imwrite('dct1dcolumn.bmp', dct1d)
    cv2.imwrite('idct1d.bmp', idct1d)

    dct1d_psnr = psnr(img, idct1d)
    print('dct1d_psnr:', dct1d_psnr)

    for cc in comp_coef:
        compressed = compress(dct1d, cc)
        for i in range(n):
            idct1d[i, :] = idct_func(compressed[i, :], c=c).squeeze()
        for i in range(n):
            idct1d[:, i] = idct_func(idct1d[:, i], c=c).squeeze()
        cv2.imwrite('idct1d_compressed_%d.bmp' % (int(1 / cc)), idct1d)
        dct1d_psnr_compressed = psnr(img, idct1d)
        print('dct1d_psnr_compressed_%d:' % (int(1 / cc)), dct1d_psnr_compressed)


def perform_dct_2d(img, dct_func, idct_func, comp_coef):
    start = time.process_time()
    n = img.shape[0]
    c = dct_matrix(n)
    dct2d = dct_func(img, c=c)
    idct2d = idct_func(dct2d, c=c)
    end = time.process_time()
    print('dct2d_time: %.2f' % (end - start))
    cv2.imwrite('dct2d.bmp', dct2d)
    cv2.imwrite('idct2d.bmp', idct2d)
    dct2d_psnr = psnr(img, idct2d)
    print('dct2d_psnr:', dct2d_psnr)

    for cc in comp_coef:
        compressed = compress(dct2d, cc)
        idct2d = idct_func(compressed, c=c)
        cv2.imwrite('idct2d_compressed_%d.bmp' % (int(1 / cc)), idct2d)
        dct2d_psnr_compressed = psnr(img, idct2d)
        print('dct2d_psnr_compressed:', dct2d_psnr_compressed)


def perform_dct_2d_block(img, dct_func, idct_func, comp_coef):
    start = time.process_time()
    c = dct_matrix(8)
    dct2d_block = blockproc(img, [8, 8], dct_func, c=c)
    idct2d_block = blockproc(dct2d_block, [8, 8], idct_func, c=c)
    end = time.process_time()
    print('dct2d_block_time: %.2f' % (end - start))
    cv2.imwrite('dct2dblock.bmp', dct2d_block)
    cv2.imwrite('idct2dblock.bmp', idct2d_block)
    dct2d_block_psnr = psnr(img, idct2d_block)
    print('dct2d_block_psnr:', dct2d_block_psnr)

    for cc in comp_coef:
        compressed = blockproc(dct2d_block, [8, 8], compress, cc)
        idct2d_block = blockproc(compressed, [8, 8], idct_func, c=c)
        cv2.imwrite('idct2d_block_compressed_%d.bmp' % (int(1 / cc)), idct2d_block)
        dct2d_block_psnr_compressed = psnr(img, idct2d_block)
        print('dct2d_block_psnr_compressed:', dct2d_block_psnr_compressed)


def main():
    # read the lena image with grayscale mode
    img = cv2.imread('lena.bmp', 0)
    cv2.imwrite('gray.bmp', img)
    img = img.astype(float)
    comp_coef = [1 / 4, 1 / 16, 1 / 64]
    perform_dct_1d(img, dct, idct, comp_coef)
    perform_dct_2d(img, dct, idct, comp_coef)
    perform_dct_2d_block(img, dct, idct, comp_coef)


if __name__ == '__main__':
    main()
