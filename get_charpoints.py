import os
import binascii
import numpy as np
import cv2
array_path = "../m5-hzk16/HZK16/Fonts/HZK16" #"../hzk-osd/font/hzk16f"# "../m5-hzk16/HZK16/Fonts/HZK16"

#检验是否含有中文字符
def is_contains_chinese(_char):
    if '\u4e00' <= _char <= '\u9fa5':
        return True


def get_arrays(ch):
    ch_gbk = ch.encode("GBK")
    # print("ch_gbk:", ch_gbk)

    ch_hex = binascii.b2a_hex(ch_gbk)
    result = str(ch_hex, encoding='utf-8')
    # L = list(ch_gbk)

    # print(L[0], L[1])
    # offset = 94*(L[0] - 0xa0 - 1) + L[1] - 0xa0 - 1

    area = eval('0x' + result[:2]) - 0xA0
    index = eval('0x' + result[2:]) - 0xA0
    offset = (94 * (area - 1) + (index - 1)) * 32

    if offset < 0:
        return None

    array_f = open(array_path, "rb")
    array_f.seek(offset)
    display = array_f.read(32)
    display = list(display)

    comp = np.array([0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01])

    column = np.zeros((16, 2))

    if len(display) < 1:
        return None

    for i in range(16):
        for j in range(2):
            column[i, j] = display[i*2+j]

    column = column[:, :, None].astype(np.byte)
    comp = comp[None, None, :].astype(np.byte)

    char_array = column & comp
    char_array = (char_array.reshape(16, 16) != 0).astype(np.int32)

    if len(np.unique(char_array)) < 2:
        return None

    '''
    for i in range(16):
        for j in range(16):
            if char_array[i, j]:
                print(char_array[i, j], end=" ")
            else:
                print("0", end=" ")
        print("\n")
    # print("char_array:", char_array)
    '''
    array_f.seek(0)

    return (char_array * 255).astype(np.uint8)


def generate_charpics(alphabet, tar_dir):

    lens = len(alphabet)

    if not os.path.isdir(tar_dir):
        os.makedirs(tar_dir)

    cnt = 1
    for ch in alphabet:
        if is_contains_chinese(ch):
            char_array = get_arrays(ch)
            if not char_array is None:
                array_path = os.path.join(tar_dir, str(cnt) + ".png")
                cv2.imwrite(array_path, char_array)
                print(cnt, "/", lens, ch)
        cnt += 1

if __name__ == "__main__":

    tar_dir = "./char_array/"
    alphabet = open("./basicsr/metrics/benchmark.txt", "r").readlines()[0]

    generate_charpics(alphabet, tar_dir)