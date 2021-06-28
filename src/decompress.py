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

from compress import zigzag, round_half

JPEG_MATRIX_QUANTIFICATION = [16, 11, 10, 16, 24, 40, 51, 61,
                              12, 12, 14, 19, 26, 58, 60, 55,
                              14, 13, 16, 24, 40, 57, 69, 56,
                              14, 17, 22, 29, 51, 87, 80, 62,
                              18, 22, 37, 56, 68, 109, 103, 77,
                              24, 35, 55, 64, 81, 104, 113, 92,
                              49, 64, 78, 87, 703, 121, 120, 101,
                              72, 92, 95, 98, 112, 100, 103, 99]


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
    res = [0] * 64
    for i in range(64):
        if i < len(m):
            res[unzig_matrix[i]] = m[i]

    return res


def unquantization(dct):
    assert len(dct) == 64
    for i in range(64):
        dct[i] *= JPEG_MATRIX_QUANTIFICATION[i]
        dct[i] = round_half(dct[i])

    return dct


def unDCT_cosine_8x8(x: float, i):
    return math.cos(
        (i * (2 * x + 1) * math.pi) / 16
    )


def unDCT_coefficients_8x8(i: int, j: int):
    # result = 1 / math.sqrt(16)
    result = 1

    if i == 0:
        result *= math.sqrt(1 / 8)
    else:
        result *= math.sqrt(2 / 8)

    if j == 0:
        result *= math.sqrt(1 / 8)
    else:
        result *= math.sqrt(2 / 8)

    return result


def unDCT_coeffs_8x8(image):
    """
    `@params`: an greyscale image (1D)

    `@returns`: generates a DCT for an image block
    """

    assert len(image) == 64
    l = [0] * 64
    range8 = range(8)
    for m, n in itertools.product(range8, range8):
        tmp = 0.0

        for u, v in itertools.product(range8, range8):
            tmp += (unDCT_cosine_8x8(n, u) *
                    unDCT_cosine_8x8(m, v) *
                    image[u * 8 + v] *
                    unDCT_coefficients_8x8(u, v))

        l[n * 8 + m] = round(tmp + 128)

    return l


if __name__ == '__main__':
    out = None
    with open("out.ourjpg", "rb") as f:
        out = f.read()

    height = out[0] * 256 + out[1]
    width = out[2] * 256 + out[3]
    padded_zeros = out[4]

    out = bytes_to_string(out[5:])[:-padded_zeros]

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

            for tmp, c in enumerate(undct):
                y, x = tmp // 8, tmp % 8

                img_y = block_y * 8 + y
                img_x = block_x * 8 + x
                print(img_y, img_x)
                # print()
                image_out[img_y][img_x][j] = c

    print(image_out)
    imageio.imwrite("test.png", image_out)


# b = (unDCT_coeffs_8x8([364, 36, 135, 32, 10, -58, -117, -85, 152, 287, 15, 0, -13, 52, 79, 46, -6, -49, -103, -250, -4, 0, 43, 45, -116, 191, -26, -59, 197, -35, 41, -
#                        26, 74, -1, -106, 78, 164, -83, -183, -90, -72, -118, 133, -201, 89, -55, 85, 189, 139, 6, 14, 58, 160, -250, 16, -36, -69, -10, 9, -41, -105, 24, 43, 193]))
# expected = [255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 0, 237, 237, 0, 255, 255, 255, 0, 185, 237, 255, 237, 255, 255, 0, 185, 185, 185,
#             237, 237, 255, 255, 0, 185, 185, 185, 185, 0, 255, 255, 0, 185, 185, 185, 0, 255, 0, 0, 255, 0, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255]

# diff = [0] * 64
# for i in range(64):
#     diff[i] = b[i] - expected[i]

# print(diff)
