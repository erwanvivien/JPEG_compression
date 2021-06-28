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

JPEG_MATRIX_QUANTIFICATION = [16, 11, 10, 16, 24, 40, 51, 61,
                              12, 12, 14, 19, 26, 58, 60, 55,
                              14, 13, 16, 24, 40, 57, 69, 56,
                              14, 17, 22, 29, 51, 87, 80, 62,
                              18, 22, 37, 56, 68, 109, 103, 77,
                              24, 35, 55, 64, 81, 104, 113, 92,
                              49, 64, 78, 87, 703, 121, 120, 101,
                              72, 92, 95, 98, 112, 100, 103, 99]
JPEG_MATRIX_QUANTIFICATION = np.array(JPEG_MATRIX_QUANTIFICATION)

ALL_COSINES = np.zeros((8, 8))
for i, j in itertools.product(range(8), range(8)):
    ALL_COSINES[i][j] = math.cos(
        (i * (2 * j + 1) * math.pi) / 16
    )


def get_dimensions(image):
    """
    `@params`: an image

    `@returns`: a tuple containing (height, width, depth=3)

    `@exceptions`: can crash if image is empty
    """
    return image.shape


def padding_8x8(image):
    """
    `@params`: an image

    `@returns`: a new image (new height / new width, multiple of 8)
    """
    y, x, depth = get_dimensions(image)

    pad_height = (8 - (y % 8)) % 8
    pad_width = (8 - (x % 8)) % 8

    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def blocks_8x8_generator(image):
    height, width, _ = get_dimensions(image)

    for y, x in itertools.product(range(0, height, 8), range(0, width, 8)):
        yield image[y:y+8, x:x+8]


def blocks_8x8(image):
    """
    `@params`: an image (height / width, multiple of 8)

    `@returns`: a list of 8 x 8 blocks
    """

    assert len(image) % 8 == 0
    assert len(image[0]) % 8 == 0

    return list(blocks_8x8_generator(image))


def DCT_coefficients_8x8(i: int, j: int):
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


def DCT_coeffs_8x8(image):
    """
    `@params`: an greyscale image (1D)

    `@returns`: generates a DCT for an image block
    """

    assert len(image) == 64

    image = image - 128
    array = np.zeros(64, dtype=np.float64)

    range8 = range(8)
    for i, j in itertools.product(range8, range8):
        idx = i * 8 + j
        for y, x in itertools.product(range8, range8):
            array[idx] += (ALL_COSINES[i][x] *
                           ALL_COSINES[j][y] *
                           image[y + x * 8])

        array[idx] *= DCT_coefficients_8x8(i, j)

    return array


def extract_channel(image, channel_name):
    """
    `@params`: an image
    `@params`: channel name ('red', 'green' or 'blue')

    `@returns`: DCT for each image block
    """
    possibles = ['red', 'green', 'blue']
    if not channel_name in possibles:
        print(f"Channel name {channel_name} does not exist")
        exit(1)

    idx = possibles.index(channel_name)

    # Keeps first 2 dimensions (height / width)
    # In the 3rd (colors) keep only the one at idx
    img = image[:, :, idx]
    return img.flatten()


def round_half(f: float):
    return int(decimal.Decimal(f).to_integral_value(rounding=decimal.ROUND_HALF_UP))


def quantization(dct):
    assert len(dct) == 64
    for i in range(64):
        dct[i] /= JPEG_MATRIX_QUANTIFICATION[i]

    return np.rint(dct).astype(np.int64)


def zigzag_generator(input_qtz):
    solution = [[] for i in range(8 + 8 - 1)]

    for i in range(8):
        for j in range(8):
            sum = i + j
            if (sum % 2 == 0):
                solution[sum].insert(0, input_qtz[i * 8 + j])
            else:
                solution[sum].append(input_qtz[i * 8 + j])

    return [e for l in solution for e in l]


def zigzag(quantized):
    assert len(quantized) == 64
    l = list(zigzag_generator(quantized))
    while l[-1] == 0:
        del l[-1]
    return l


def huffman_category(n):
    absN = abs(n)
    if absN <= 1:
        return absN

    inf, sup = 2, 4
    for i in range(1, 16):
        if absN >= inf and absN < sup:
            return i + 1
        inf, sup = sup, sup * 2


def huffman_number(n):
    if n >= 0:
        return n

    absN = -n
    inf, sup = 0, 1
    for i in range(-1, 16):
        if absN >= inf and absN < sup:
            break
        inf, sup = sup, sup * 2

    n += (sup - 1)
    return n


def huffman_DC_encoding(n):
    cat = huffman_category(n)
    vals = DCTable[cat]
    basecode, shift, length = vals["basecode"], vals["shift"], vals["length"]

    s = f"{basecode << shift | huffman_number(n):b}"
    s = ("0" * (length - len(s))) + s

    return s


def huffman_AC_encoding(n, zeros=0):
    cat = huffman_category(n)
    vals = ACTable[zeros][cat]
    basecode, shift, length = vals["basecode"], vals["shift"], vals["length"]

    s = f"{basecode << shift | huffman_number(n):b}"
    s = ("0" * (length - len(s))) + s

    return s


def parse_zigzag(zigzag):
    DC = huffman_DC_encoding(zigzag.pop(0))
    ACs = []

    while zigzag:
        zeros = 0
        while zigzag[0] == 0 and zeros < 15:
            zigzag.pop(0)
            zeros += 1
        ACs.append(huffman_AC_encoding(zigzag.pop(0), zeros))

    ACs.append("1010")

    ACs_ = " ".join(ACs)
    return f"{DC} {ACs_}"


def string_to_bytes(s):
    bs = []
    for i in range(0, len(s), 8):
        tmp = 0
        for j in range(8):
            tmp += int(s[i + j]) * (2 ** (7 - j))
        bs.append(tmp)
    return bs


def compress(filename):
    image = imageio.imread(filename)
    image = image.astype(np.int16)

    padded_img = padding_8x8(image)
    blocks = blocks_8x8(padded_img)
    colors = ["red", "green", "blue"]

    res = ""
    for b in blocks:
        for color in colors:
            cha = extract_channel(b, color)
            dct = DCT_coeffs_8x8(cha)
            qtz = quantization(dct)
            zzg = zigzag(qtz)
            out = parse_zigzag(zzg)
            res += out

    res = res.replace(" ", "")

    height, width, _ = get_dimensions(image)
    padded_zeros = (8 - (len(res) % 8)) % 8

    res += "0" * padded_zeros
    bytes_to_write = [height // 256, height % 256,
                      width // 256, width % 256,
                      padded_zeros]

    bytes_to_write = bytes(bytes_to_write + string_to_bytes(res))
    with open("out.ourjpg", "wb") as f:
        f.write(bytes_to_write)


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if argv:
        compress(argv[0])
    else:
        compress("assets/256.jpg")
