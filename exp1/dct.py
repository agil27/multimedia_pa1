import cv2
import numpy as np
import time
from utils import *
import os


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

    cv2.imwrite('output/dct1drow.bmp', dct1drow)
    cv2.imwrite('output/dct1dcolumn.bmp', dct1d)
    cv2.imwrite('output/idct1d.bmp', idct1d)

    dct1d_psnr = psnr(img, idct1d)
    print('dct1d_psnr:', dct1d_psnr)

    for cc in comp_coef:
        compressed = compress(dct1d, cc)
        for i in range(n):
            idct1d[i, :] = idct_func(compressed[i, :], c=c).squeeze()
        for i in range(n):
            idct1d[:, i] = idct_func(idct1d[:, i], c=c).squeeze()
        cv2.imwrite('output/idct1d_compressed_%d.bmp' % (int(1 / cc)), idct1d)
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
    cv2.imwrite('output/dct2d.bmp', dct2d)
    cv2.imwrite('output/idct2d.bmp', idct2d)
    dct2d_psnr = psnr(img, idct2d)
    print('dct2d_psnr:', dct2d_psnr)

    for cc in comp_coef:
        compressed = compress(dct2d, cc)
        idct2d = idct_func(compressed, c=c)
        cv2.imwrite('output/idct2d_compressed_%d.bmp' % (int(1 / cc)), idct2d)
        dct2d_psnr_compressed = psnr(img, idct2d)
        print('dct2d_psnr_compressed:', dct2d_psnr_compressed)


def perform_dct_2d_block(img, dct_func, idct_func, comp_coef):
    start = time.process_time()
    c = dct_matrix(8)
    dct2d_block = blockproc(img, [8, 8], dct_func, c=c)
    idct2d_block = blockproc(dct2d_block, [8, 8], idct_func, c=c)
    end = time.process_time()
    print('dct2d_block_time: %.2f' % (end - start))
    cv2.imwrite('output/dct2dblock.bmp', dct2d_block)
    cv2.imwrite('output/idct2dblock.bmp', idct2d_block)
    dct2d_block_psnr = psnr(img, idct2d_block)
    print('dct2d_block_psnr:', dct2d_block_psnr)

    for cc in comp_coef:
        compressed = blockproc(dct2d_block, [8, 8], compress, cc)
        idct2d_block = blockproc(compressed, [8, 8], idct_func, c=c)
        cv2.imwrite('output/idct2d_block_compressed_%d.bmp' % (int(1 / cc)), idct2d_block)
        dct2d_block_psnr_compressed = psnr(img, idct2d_block)
        print('dct2d_block_psnr_compressed:', dct2d_block_psnr_compressed)


def main():
    if not os.path.exists('output'):
        os.makedirs('output')
    # read the lena image with grayscale mode
    img = cv2.imread('lena.bmp', 0)
    cv2.imwrite('output/gray.bmp', img)
    img = img.astype(float)
    comp_coef = [1 / 4, 1 / 16, 1 / 64]
    perform_dct_1d(img, dct, idct, comp_coef)
    perform_dct_2d(img, dct, idct, comp_coef)
    perform_dct_2d_block(img, dct, idct, comp_coef)


if __name__ == '__main__':
    main()

"""
output result
dct1d_time: 1.20
dct1d_psnr: 274.4943998392947
dct1d_psnr_compressed_4: 36.17668334284461
dct1d_psnr_compressed_16: 29.743323739686257
dct1d_psnr_compressed_64: 25.97893592965811
dct2d_time: 0.80
dct2d_psnr: 274.51900930900416
dct2d_psnr_compressed: 36.17668334284461
dct2d_psnr_compressed: 29.743323739686257
dct2d_psnr_compressed: 25.97893592965811
dct2d_block_time: 0.09
dct2d_block_psnr: 310.3364673595838
dct2d_block_psnr_compressed: 34.782208938976574
dct2d_block_psnr_compressed: 27.68377691902231
dct2d_block_psnr_compressed: 23.853019450907777
"""
