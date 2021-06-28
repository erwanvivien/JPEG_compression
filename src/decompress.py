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


def huffmand_decode(b):
    pass


if __name__ == '__main__':
    out = None
    with open("out.ourjpg", "rb") as f:
        out = f.read()

    height = out[0] * 256 + out[1]
    width = out[2] * 256 + out[3]
    padded_zeros = out[4]

    out = bytes_to_string(out[5:])[:-padded_zeros]
