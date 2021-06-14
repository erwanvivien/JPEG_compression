import os
import imageio

import numpy as np
import itertools
import math

import decimal


JPEG_MATRIX_QUANTIFICATION = [16, 11, 10, 16, 24, 40, 51, 61,
                              12, 12, 14, 19, 26, 58, 60, 55,
                              14, 13, 16, 24, 40, 57, 69, 56,
                              14, 17, 22, 29, 51, 87, 80, 62,
                              18, 22, 37, 56, 68, 109, 103, 77,
                              24, 35, 55, 64, 81, 104, 113, 92,
                              49, 64, 78, 87, 703, 121, 120, 101,
                              72, 92, 95, 98, 112, 100, 103, 99]


def get_dimensions(image):
    """
    `@params`: an image

    `@returns`: a tuple containing (height, width, depth=3)

    `@exceptions`: can crash if image is empty
    """
    y = len(image)
    if y <= 0:
        print(f"Image is empty")
        exit(1)

    x = len(image[0])
    if x <= 0:
        print(f"Image is empty")
        exit(1)

    depth = len(image[0][0])
    if x <= 0:
        print(f"Image is empty")
        exit(1)

    return y, x, depth


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


def DCT_cosine_8x8(x: float, i):
    return math.cos(
        (i * (2 * x + 1) * math.pi) / 16
    )


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


def DCT_coeffs_8x8_generator(image):
    range8 = range(8)
    for i, j in itertools.product(range8, range8):
        tmp = 0.0

        for y, x in itertools.product(range8, range8):
            tmp += (DCT_cosine_8x8(x, i) *
                    DCT_cosine_8x8(y, j) *
                    (image[y + x * 8] - 128))

        tmp *= DCT_coefficients_8x8(i, j)

        yield round(tmp)


def DCT_coeffs_8x8(image):
    """
    `@params`: an greyscale image (1D)

    `@returns`: generates a DCT for an image block
    """

    assert len(image) == 64
    return list(get_dct_coefs_8x8_generator(image))


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
    return image[:, :, idx]


def round_half(f: float):
    return decimal.Decimal(f).to_integral_value(rounding=decimal.ROUND_HALF_UP)


def quantization(dct):
    assert len(dct) == 64
    for i in range(64):
        dct[i] /= JPEG_MATRIX_QUANTIFICATION[i]
        dct[i] = round_half(dct[i])

    return dct


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


if __name__ == "__main__":
    pass
    # filename: str = 'color.jpg'

    # image = imageio.imread(filename)
    # height, width, depth = get_dimensions(image)

    # padded_img = padding_8x8(image)
    # blocks = blocks_8x8(padded_img)

    # test_block = extract_channel(blocks[6], "green")
    # imageio.imwrite("out.jpg", test_block)


def test_dct_1():
    input_pixel_matrix = [140, 144, 147, 140, 140, 155, 179, 175,
                          144, 152, 140, 147, 140, 148, 167, 179,
                          152, 155, 136, 167, 163, 162, 152, 172,
                          168, 145, 156, 160, 152, 155, 136, 160,
                          162, 148, 156, 148, 140, 136, 147, 162,
                          147, 167, 140, 155, 155, 140, 136, 162,
                          136, 156, 123, 167, 162, 144, 140, 147,
                          148, 155, 136, 155, 152, 147, 147, 136]

    output_pixel_matrix_expected = [186, -18, 15, -9, 23, -9, -14, -19,
                                    21, -34, 26, -9, -11, 11, 14, 7,
                                    -10, -24, -2, 6, -18, 3, -20, -1,
                                    -8, -5, 14, -15, -8, -3, -3, 8,
                                    -3, 10, 8, 1, -11, 18, 18, 15,
                                    4, -2, -18, 8, 8, -4, 1, -7,
                                    9, 1, -3, 4, -1, -7, -1, -2,
                                    0, -8, -2, 2, 1, 4, -6, 0]

    output_pixel_matrix = DCT_coeffs_8x8(input_pixel_matrix)

    for i in range(64):
        if output_pixel_matrix[i] != output_pixel_matrix_expected[i]:
            print("test_dct_1", i,
                  output_pixel_matrix[i], output_pixel_matrix_expected[i])


def test_division_wise_1():
    input_dct = [
        -415, -30, -61, 27, 56, -20, -2, 0,
        4, -22, -61, 10, 13, -7, -9, 5,
        -47, 7, 77, -25, -29, 10, 5, -6,
        -49, 12, 34, -15, -10, 6, 2, 2,
        12, -7, -13, -4, -2, 2, -3, 3,
        -8, 3, 2, -6, -2, 1, 4, 2,
        -1, 0, 0, -2, -1, -3, 4, -1,
        0, 0, -1, -4, -1, 0, 1, 2]

    output_expected = [
        -26, -3, -6, 2, 2, -1, 0, 0,
        0, -2, -4, 1, 1, 0, 0, 0,
        -3, 1, 5, -1, -1, 0, 0, 0,
        -3, 1, 2, -1, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]

    output = quantization(input_dct)
    for i in range(64):
        if output[i] != output_expected[i]:
            print("test_division_wise_1:", i, "\n",
                  output[i],  output_expected[i])


def test_zigzag():
    input_qtz = [
        -26, -3, -6, 2, 2, -1, 0, 0,
        0, -2, -4, 1, 1, 0, 0, 0,
        -3, 1, 5, -1, -1, 0, 0, 0,
        -3, 1, 2, -1, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]

    print(zigzag(input_qtz))


test_zigzag()
