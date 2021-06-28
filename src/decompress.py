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
    print([DC] + ACs)
    return [DC] + ACs


def huffman_decode(b, height, width):
    # Offset is always a list, so it's like pointer
    offset = [0]

    for i in range(math.ceil(height / 8)):
        for j in range(math.ceil(width / 8)):
            for _ in range(3):
                huffman_decode_block(b, offset)

    return


if __name__ == '__main__':
    out = None
    with open("out.ourjpg", "rb") as f:
        out = f.read()

    height = out[0] * 256 + out[1]
    width = out[2] * 256 + out[3]
    padded_zeros = out[4]

    out = bytes_to_string(out[5:])[:-padded_zeros]

    huffman_decode(out, height, width)
