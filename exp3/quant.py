from utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

Q = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]]
)

CANNON = np.array(
    [[1, 1, 1, 2, 3, 6, 8, 10],
     [1, 1, 2, 3, 4, 8, 9, 8],
     [2, 2, 2, 3, 6, 8, 10, 8],
     [2, 2, 3, 4, 7, 12, 11, 9],
     [3, 3, 8, 11, 10, 16, 15, 11],
     [3, 5, 8, 10, 12, 15, 16, 13],
     [7, 10, 11, 12, 15, 17, 17, 14],
     [14, 13, 13, 15, 15, 14, 14, 14]]
)

NIKON = np.array(
    [[2, 1, 1, 2, 3, 5, 6, 7],
     [1, 1, 2, 2, 3, 7, 7, 7],
     [2, 2, 2, 3, 5, 7, 8, 7],
     [2, 2, 3, 3, 6, 10, 10, 7],
     [2, 3, 4, 7, 8, 13, 12, 9],
     [3, 4, 7, 8, 10, 12, 14, 11],
     [6, 8, 9, 10, 12, 15, 14, 12],
     [9, 11, 11, 12, 13, 12, 12, 12]]
)

ZIGZAG = np.array(
    [[1, 1, 2, 2, 4, 4, 7, 8],
     [1, 2, 3, 4, 5, 7, 8, 11],
     [1, 3, 4, 5, 7, 8, 11, 12],
     [3, 3, 5, 7, 9, 11, 12, 14],
     [3, 6, 6, 9, 10, 12, 14, 14],
     [6, 6, 9, 10, 12, 13, 15, 16],
     [6, 9, 10, 12, 13, 15, 15, 16],
     [9, 10, 13, 13, 15, 15, 16, 17]]
)


def quant(x, q, c, meter):
    global psnr_meter
    dct2d = dct(x, c=c)
    quant_dct = np.round(dct2d / q)
    idct2d = idct(quant_dct * q, c=c)
    meter.append(psnr(x, idct2d))
    return idct2d


def block_quant(x, q, block, c=None):
    if c is None:
        c = dct_matrix(block[0])
    psnr = AverageMeter()
    blockproc(x, block, quant, q, c, psnr)
    return x, psnr


def main():
    img = cv2.imread('lena.bmp', 0)
    img = img.astype(float)
    for q, qname in [(Q, 'JPEG'), (CANNON, 'CANNON'), (NIKON, 'NIKON'), (ZIGZAG, 'ZIGZAG')]:
        if not os.path.exists(os.path.join('output', qname)):
            os.makedirs(os.path.join('output', qname))
        c = dct_matrix(8)
        _, psnr_data = block_quant(img, q, [8, 8], c=c)
        psnr_data.log('output/block_info.txt', '--------%s--------' % qname)
        qmeter = AverageMeter()
        for i in range(1, 21):
            a = float(i) / 10
            quantized, psnr_data = block_quant(img, q * a, [8, 8], c=c)
            cv2.imwrite('output/%s/a_%.1f.bmp' % (qname, a), quantized)
            qmeter.append(psnr_data.mean())

        x = np.arange(1, 21) / 10
        y = np.array(qmeter.content())
        plt.title('%s PSNR' % 'lena')
        plt.xlabel('Parameter a')
        plt.ylabel('Average PSNR')
        plt.plot(x, y, label=qname)

    plt.legend()
    plt.savefig('output/psnr.png')


if __name__ == '__main__':
    main()
