import os
import imageio

import numpy as np
import itertools
import math


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


def get_dct_coefs_8x8_generator():
    # using
    # https://www-ljk.imag.fr/membres/Valerie.Perrier/SiteWeb/node9.html
    sqrt_1_n = 1 / math.sqrt(8)
    sqrt_2_n = math.sqrt(2 / 8)

    for i, j in itertools.product(range(8), range(8)):
        if i == 0:
            yield (sqrt_1_n)
        elif i > 0:
            yield (sqrt_2_n * math.cos(
                (2*j + 1) * (i * math.pi) / 16
            ))


def get_dct_coefs_8x8(height=8, width=8):
    """
    `@params`: a width and height

    `@returns`: DCT for each image block
    """

    return np.array(list(get_dct_coefs_8x8_generator()))


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


if __name__ == "__main__":
    filename: str = 'color.jpg'
    image = imageio.imread(filename)

    height, width, depth = get_dimensions(image)

    padded_img = padding_8x8(image)
    blocks = blocks_8x8(padded_img)

    C = get_dct_coefs_8x8()
    Ct = np.matrix.transpose(C)

    # test_block = extract_channel(blocks[6], "red")
    # imageio.imwrite("out.jpg", test_block)
