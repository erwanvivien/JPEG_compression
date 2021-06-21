
DCTable = [
    ("010", 3), ("011", 4), ("100", 5), ("00", 5),
    ("101", 7), ("110", 8), ("1110", 10),
    ("11110", 12), ("111110", 14), ("1111110", 16),
    ("11111110", 18), ("111111110", 20)
]

DCTable = [{"basecode": int(e[0], 2),
            "length": e[1],
            "shift": e[1] - len(e[0])} for e in DCTable]


ACTable = [
    [  # 0/0 - 0/A
        ("1010", 4),
        ("00", 3),
        ("01", 4),
        ("100", 6),
        ("1011", 8),
        ("11010", 10),
        ("111000", 12),
        ("1111000", 14),
        ("1111110110", 18),
        ("1111111110000010", 25),
        ("1111111110000011", 26),
    ],
    [  # 1/1 - 1/A
        ("0", -1),
        ("1100", 5),
        ("111001", 8),
        ("1111001", 10),
        ("111110110", 13),
        ("11111110110", 16),
        ("1111111110000100", 22),
        ("1111111110000101", 23),
        ("1111111110000110", 24),
        ("1111111110000111", 25),
        ("1111111110001000", 26),
    ],
    [  # 2/1 - 2/A
        ("0", -1),
        ("11011", 6),
        ("11111000", 10),
        ("1111110111", 13),
        ("1111111110001001", 20),
        ("1111111110001010", 21),
        ("1111111110001011", 22),
        ("1111111110001100", 23),
        ("1111111110001101", 24),
        ("1111111110001110", 25),
        ("1111111110001111", 26),
    ],
    [  # 3/1 - 3/A
        ("0", -1),
        ("111010", 7),
        ("111110111", 11),
        ("11111110111", 14),
        ("1111111110010000", 20),
        ("1111111110010001", 21),
        ("1111111110010010", 22),
        ("1111111110010011", 23),
        ("1111111110010100", 24),
        ("1111111110010101", 25),
        ("1111111110010110", 26),
    ],
    [  # 4/1 - 4/A
        ("0", -1),
        ("111011", 7),
        ("1111111000", 12),
        ("1111111110010111", 19),
        ("1111111110011000", 20),
        ("1111111110011001", 21),
        ("1111111110011010", 22),
        ("1111111110011011", 23),
        ("1111111110011100", 24),
        ("1111111110011101", 25),
        ("1111111110011110", 26),
    ],
    [  # 5/1 - 5/A
        ("0", -1),
        ("1111010", 7),
        ("1111111001", 12),
        ("1111111110011111", 19),
        ("1111111110100000", 20),
        ("1111111110100001", 21),
        ("1111111110100010", 22),
        ("1111111110100011", 23),
        ("1111111110100100", 24),
        ("1111111110100101", 25),
        ("1111111110100110", 26),
    ],
    [  # 6/1 - 6/A
        ("0", -1),
        ("1111011", 8),
        ("11111111000", 13),
        ("1111111110100111", 19),
        ("1111111110101000", 20),
        ("1111111110101001", 21),
        ("1111111110101010", 22),
        ("1111111110101011", 23),
        ("1111111110101100", 24),
        ("1111111110101101", 25),
        ("1111111110101110", 26),
    ],
    [  # 7/1 - 7/A
        ("0", -1),
        ("1111011", 9),
        ("11111111000", 13),
        ("1111111110100111", 19),
        ("1111111110101000", 20),
        ("1111111110101001", 21),
        ("1111111110101010", 22),
        ("1111111110101011", 23),
        ("1111111110101100", 24),
        ("1111111110101101", 25),
        ("1111111110101110", 26),
    ],
    [  # 8/1 - 8/A
        ("0", -1),
        ("11111010", 9),
        ("111111111000000", 17),
        ("1111111110110111", 19),
        ("1111111110111000", 20),
        ("1111111110111001", 21),
        ("1111111110111010", 22),
        ("1111111110111011", 23),
        ("1111111110111100", 24),
        ("1111111110111101", 25),
        ("1111111110111110", 26),
    ],
    [  # 9/1 - 9/A
        ("0", -1),
        ("111111000", 10),
        ("1111111110111111", 18),
        ("1111111111000000", 19),
        ("1111111111000001", 20),
        ("1111111111000010", 21),
        ("1111111111000011", 22),
        ("1111111111000100", 23),
        ("1111111111000101", 24),
        ("1111111111000110", 25),
        ("1111111111000111", 26),
    ],
    [  # A/1 - A/A
        ("0", -1),
        ("111111001", 10),
        ("1111111111001000", 18),
        ("1111111111001001", 19),
        ("1111111111001010", 20),
        ("1111111111001011", 21),
        ("1111111111001100", 22),
        ("1111111111001101", 23),
        ("1111111111001110", 24),
        ("1111111111001111", 25),
        ("1111111111010000", 26),
    ],
    [  # B/1 - B/A
        ("0", -1),
        ("111111010", 10),
        ("1111111111010001", 18),
        ("1111111111010010", 19),
        ("1111111111010011", 20),
        ("1111111111010100", 21),
        ("1111111111010101", 22),
        ("1111111111010110", 23),
        ("1111111111010111", 24),
        ("1111111111011000", 25),
        ("1111111111011001", 26),
    ],
    [  # C/1 - C/A
        ("0", -1),
        ("1111111010", 11),
        ("1111111111011010", 18),
        ("1111111111011011", 19),
        ("1111111111011100", 20),
        ("1111111111011101", 21),
        ("1111111111011110", 22),
        ("1111111111011111", 23),
        ("1111111111100000", 24),
        ("1111111111100001", 25),
        ("1111111111100010", 26),
    ],
    [  # D/1 - D/A
        ("0", -1),
        ("11111111010", 12),
        ("1111111111100011", 18),
        ("1111111111100100", 19),
        ("1111111111100101", 20),
        ("1111111111100110", 21),
        ("1111111111100111", 22),
        ("1111111111101000", 23),
        ("1111111111101001", 24),
        ("1111111111101010", 25),
        ("1111111111101011", 26),
    ],
    [  # E/1 - E/A
        ("0", -1),
        ("111111110110", 7),
        ("1111111111101100", 18),
        ("1111111111101101", 19),
        ("1111111111101110", 20),
        ("1111111111101111", 21),
        ("1111111111110000", 22),
        ("1111111111110001", 23),
        ("1111111111110010", 24),
        ("1111111111110011", 25),
        ("1111111111110100", 26),
    ],
    [  # F/0 - F/A
        ("111111110111", 12),  # F/0 => 16 zeros d'affilés
        ("1111111111110101", 17),
        ("1111111111110110", 18),
        ("1111111111110111", 19),
        ("1111111111111000", 20),
        ("1111111111111001", 21),
        ("1111111111111010", 22),
        ("1111111111111011", 23),
        ("1111111111111100", 24),
        ("1111111111111101", 25),
        ("1111111111111110", 26),
    ]
]


for i, elem in enumerate(ACTable):
    tmp = [{"basecode": int(e[0], 2),
            "length": e[1],
            "shift": e[1] - len(e[0])} for e in elem]
    ACTable[i] = tmp