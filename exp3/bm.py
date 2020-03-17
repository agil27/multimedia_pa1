import cv2
import numpy as np
from utils import *
from quant import *


FIRST_BLOCK = [165, 201]
MAX_MAD = 1000000


def mad(img, last, block, mv):
    h, w = img.shape[0], img.shape[1]
    m, n = block[0], block[1]
    i, j = mv[0], mv[1]
    if m + i < 0 or m + i + 16 >= h or n + j < 0 or n + j + 16 >= w:
        return MAX_MAD
    old_view = last[m : m + 16, n :  n + 16]
    new_view = img[m + i : m + i + 16, n + j : n + j + 16]
    # diff = np.abs(new_view - old_view)
    # mse = np.mean(diff ** 2)
    # mae = np.mean(diff)
    nccf = np.sum(new_view * old_view) / ((np.sqrt(np.sum(new_view ** 2))) * (np.sqrt(np.sum(old_view ** 2))))
    return nccf


def search_around(s, img, last, block, init_mv):
    mvs = [(-s, -s), (-s, 0), (0, -s), (0, 0), (-s, s), (s, -s), (0, s), (s, 0), (s, s)]
    min_mad = MAX_MAD
    min_mv = (0, 0)
    for mv in mvs:
        curr_mv = (init_mv[0] + mv[0], init_mv[1] + mv[1])
        curr_mad = mad(img, last, block, curr_mv)
        if curr_mad < min_mad:
            min_mad = curr_mad
            min_mv = (curr_mv[0], curr_mv[1])
    return min_mad, min_mv


def tss(img, last, block):
    s = 4
    min_mad, min_mv = search_around(s, img, last, block, (0, 0))
    s = 2
    min_mad, min_mv = search_around(s, img, last, block, min_mv)
    s = 1
    return search_around(s, img, last, block, min_mv)


def gs(img, last, block):
    min_mad = MAX_MAD
    min_mv = (0, 0)
    for i in range(0, 15):
        for j in range(0, 15):
            mv = (i - 7, j - 7)
            curr_mad = mad(img, last, block, mv)
            # if curr_mad < min_mad and mv != (0, 0):
            if curr_mad < min_mad:
                min_mad = curr_mad
                min_mv = mv
    return min_mad, min_mv


def pixel_bm():
    cap = cv2.VideoCapture()
    cap.open('cars.avi')

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    block = FIRST_BLOCK
    last_frame = None

    for i in range(frames):
        _, frame = cap.read()
        draw = cv2.rectangle(frame, (block[1], block[0]), (block[1] + 16, block[0] + 16), (0, 255, 0), 2)
        cv2.imshow('draw', draw)
        cv2.waitKey(10)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(float)
        if i > 0:
            mad, mv = gs(frame, last_frame, block)
            block = (block[0] + mv[0], block[1] + mv[1])
            if block[0] == 0 or block[1] == 0 or block[0] + 16 == frame.shape[0] - 1 or block[1] + 16 == frame.shape[0] - 1:
                block = FIRST_BLOCK
        last_frame = frame


def compression_bm():
    cap = cv2.VideoCapture()
    cap.open('cars.avi')

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    block = FIRST_BLOCK
    last_frame = None
    c = dct_matrix(8)

    for i in range(frames):
        _, frame = cap.read()
        draw = cv2.rectangle(frame, (block[1], block[0]), (block[1] + 16, block[0] + 16), (0, 255, 0), 2)
        cv2.imshow('draw', draw)
        cv2.waitKey(10)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(float)
        frame = blockproc(frame, [8, 8], dct, c=c)
        frame = blockproc(frame, [8, 8], idct, c=c)
        if i > 0:
            mad, mv = gs(frame, last_frame, block)
            block = (block[0] + mv[0], block[1] + mv[1])
            if block[0] == 0 or block[1] == 0 or block[0] + 16 == frame.shape[0] - 1 or block[1] + 16 == frame.shape[0] - 1:
                block = FIRST_BLOCK
        last_frame = frame


def main():
    # pixel_bm()
    compression_bm()


if __name__ == '__main__':
    main()
