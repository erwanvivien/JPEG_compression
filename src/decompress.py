import os
import imageio

import numpy as np
import itertools
import math

import decimal
import unittest

import random

from tables import DCTable
from tables import ACTable
from tables import invDCTable
from tables import invACTable

from compress import zigzag
from compress import ALL_COSINES
from compress import DCT
from compress import DCT_t


JPEG_MATRIX_QUANTIFICATION = [16, 11, 10, 16, 24, 40, 51, 61,
                              12, 12, 14, 19, 26, 58, 60, 55,
                              14, 13, 16, 24, 40, 57, 69, 56,
                              14, 17, 22, 29, 51, 87, 80, 62,
                              18, 22, 37, 56, 68, 109, 103, 77,
                              24, 35, 55, 64, 81, 104, 113, 92,
                              49, 64, 78, 87, 703, 121, 120, 101,
                              72, 92, 95, 98, 112, 100, 103, 99]
JPEG_MATRIX_QUANTIFICATION = np.array(JPEG_MATRIX_QUANTIFICATION)


def bytes_to_string(b):
    out = ""
    for i in b:
        for j in range(8):
            offset = 2 ** (7 - j)
            if i & offset:
                out += "1"
            else:
                out += "0"
    return out


def read_char(b, offset):
    res = b[offset[0]]
    offset[0] += 1
    return res


def search_in(b, offset, dico):
    search = read_char(b, offset)

    while search not in dico and len(search) < 100:
        search += read_char(b, offset)

    if search not in dico:
        print("huffman_decode_DC: Big error")
        exit(1)

    return dico[search]


def huffman_number_2(s, cat):
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return -1 if s == "0" else 1
    if s[0] == '0':
        return -2 ** (cat) + 1 + (int(s[1:], 2)) if s[1:] else 0
    else:
        return 2 ** (cat - 1) + (int(s[1:], 2)) if s[1:] else 0


def huffman_decode_DC(b, offset):
    table = search_in(b, offset, invDCTable)
    nb_to_read = table["to_read"]

    res = ""
    for _ in range(nb_to_read):
        res += read_char(b, offset)

    return huffman_number_2(res, table["cat"])


def huffman_decode_AC(b, offset):
    res = []
    while True:
        table = search_in(b, offset, invACTable)
        nb_to_read = table["to_read"]
        if nb_to_read == 0 and table["length"] == 4:
            return res

        res_str = ""
        for _ in range(nb_to_read):
            res_str += read_char(b, offset)

        for _ in range(table["run"]):
            res.append(0)

        if nb_to_read != 0:
            res.append(huffman_number_2(res_str, table["cat"]))
        else:
            res.append(0)

    return res


def huffman_decode_block(b, offset):
    DC = huffman_decode_DC(b, offset)
    ACs = huffman_decode_AC(b, offset)
    return [DC] + ACs


def huffman_decode_generator(b, height, width):
    # Offset is always a list, so it's like pointer
    offset = [0]

    for i in range(math.ceil(height / 8)):
        for j in range(math.ceil(width / 8)):
            for _ in range(3):
                yield huffman_decode_block(b, offset)


def huffman_decode(b, height, width):
    return list(huffman_decode_generator(b, height, width))


unzig_matrix = zigzag(list(range(64)))


def unzigzag(m):
    res = np.zeros(64)
    for i in range(64):
        if i < len(m):
            res[unzig_matrix[i]] = m[i]

    return res


def unquantization(dct):
    assert len(dct) == 64
    for i in range(64):
        dct[i] *= JPEG_MATRIX_QUANTIFICATION[i]

    return np.rint(dct).astype(np.int64)


def unDCT_coeffs_8x8(image):
    image = image.reshape((8, 8))
    res = np.dot(np.dot(DCT_t, image), DCT)
    return res.reshape(64).clip(0, 255)


def decompress(filename):
    global JPEG_MATRIX_QUANTIFICATION
    out = None
    with open(filename, "rb") as f:
        out = f.read()

    height = out[0] * 256 + out[1]
    width = out[2] * 256 + out[3]
    padded_zeros = out[4]
    q = out[5]
    if q != 50:
        if q > 50:
            alpha = 200 - 2 * q
        else:
            alpha = 5000 / q
        JPEG_MATRIX_QUANTIFICATION = (
            (alpha * JPEG_MATRIX_QUANTIFICATION) + 50) / 100

    out = bytes_to_string(out[6:])
    if padded_zeros > 0:
        out = out[:-padded_zeros]

    pad_height = (8 - (height % 8)) % 8
    pad_width = (8 - (width % 8)) % 8

    virtual_height = height + pad_height
    virtual_width = width + pad_width

    image_out = np.zeros((virtual_height, virtual_width, 3), dtype=np.uint8)

    block_width = virtual_width // 8

    blocks = huffman_decode(out, height, width)

    for i in range(0, len(blocks), 3):
        for j in [0, 1, 2]:
            b = blocks[i + j]
            block_y = (i // 3) // block_width
            block_x = (i // 3) % block_width

            unzzg = unzigzag(b)
            unqtz = unquantization(unzzg)
            undct = unDCT_coeffs_8x8(unqtz)

            reshp = undct.reshape((8, 8))
            image_out[block_y * 8:block_y * 8 + 8, block_x * 8:block_x * 8+8, j] = \
                reshp

    imageio.imwrite("test.png", image_out)


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if argv:
        decompress(argv[0])
    else:
        decompress("out.ourjpg")
