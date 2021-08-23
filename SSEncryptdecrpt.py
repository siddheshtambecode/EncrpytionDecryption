import math

import pandas as pd
import numpy as np

'''
Encrpytion and decryption using ______ and ______ together


__author__ = SIDDHESH TAMBE
__github__ = https://github.com/siddheshtambecode
'''


def main():
    '''  Enter Keyword     '''
    keyword = 'holo'

    ''' Enter plain text Pi'''
    plain_text = 'siddhesh'

    '''2nd key'''
    second_key = "qcuh"
    second_key = second_key.lower()

    print(" Plain text  ", plain_text)
    print("First Key ", keyword)
    print("Second key ", second_key)
    '''
    Defining dictionary 
    '''
    alphabet_dict = {}
    eng_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i, letter in enumerate(eng_alphabet):
        alphabet_dict[letter] = i
    print(alphabet_dict)

    keyword = keyword.lower()
    plain_text = plain_text.lower()
    if len(keyword) > len(plain_text):
        print('Incorrect input.Exiting...')
        exit()

    '''Reverse the string Ri'''
    reversed_plain_text = plain_text[::-1]

    '''ti = Ri+ Ki
       if (Ri.length > ki.length)
       then repeat the ki = Ri.length
       '''

    Ti = generate_ti(keyword, reversed_plain_text, alphabet_dict)
    print("Ti : ", Ti)

    Ci = generate_ci(Ti, alphabet_dict)

    print("Ci :", Ci)

    matrix = generate_matrix_encrypt(second_key, Ci, alphabet_dict)

    print(matrix)

    encrypted_final = get_final_encrpytion(matrix, alphabet_dict)

    print("Final Encrypted Text", encrypted_final)

    print("Completed Encryption ")

    print("Begining decryption..........")

    decrpytion_matrix_str = get_dedcryption_matrix_str(encrypted_final, second_key, alphabet_dict)

    print("Decryption matrix string " + decrpytion_matrix_str)

    decrpytion_str_array = get_decryption_str_array(decrpytion_matrix_str, alphabet_dict, keyword)

    print("Getting final decrypted string")
    final_decrypted_str = get_final_decryption(decrpytion_str_array, alphabet_dict)

    print("Final decrypted string " + final_decrypted_str)


def generate_keyword_pad(reversed_plain_text, keyword):
    keyword_pad = ""
    repeat_times = int((len(reversed_plain_text) / len(keyword)))
    last_n_chars = len(reversed_plain_text) % len(keyword)
    count = 0
    while count < repeat_times:
        keyword_pad = keyword_pad + keyword
        count = count + 1
    keyword_pad = keyword_pad + keyword[0:last_n_chars]
    print("Keyword_pad " + keyword_pad)
    return keyword_pad


def generate_ti(keyword, reversed_plain_text, alphabet_dict):
    ti = []
    ti_keyword_pad = []
    ti_reversed_plain_text = []
    if len(keyword) == len(reversed_plain_text):
        ti = keyword
    if len(keyword) < len(reversed_plain_text):
        '''Calculate keyword pad'''
        keyword_pad = generate_keyword_pad(reversed_plain_text, keyword)
    for char in keyword_pad:
        ti_keyword_pad.append(alphabet_dict[char])
    for char in reversed_plain_text:
        ti_reversed_plain_text.append(alphabet_dict[char])
    print(ti_reversed_plain_text)
    print(ti_keyword_pad)
    assert len(ti_reversed_plain_text) == len(ti_keyword_pad)
    ti = [a + b for a, b in zip(ti_keyword_pad, ti_reversed_plain_text)]
    return ti


def generate_ci(Ti, alphabet_dict):
    Ci = ""
    for char in Ti:
        char = char % 26
        Ci = Ci + str([k for k, v in alphabet_dict.items() if v == char][0])
        # print("Char: ", Ci , "  " ,char)
    return Ci


def generate_matrix_encrypt(second_key, Ci, alphabet_dict):
    rows = math.ceil(len(Ci) / len(second_key)) + 1
    cols = len(second_key)
    print("Matrix rows ", rows)
    print("Matrix cols", cols)
    matrix = np.zeros((rows, cols), dtype=object)
    # matrix = [len(Ci)//len(second_key)][len(second_key)]
    print("Matrix Shape ", matrix.shape)
    header_array = []
    for char in second_key:
        header_array.append(int(alphabet_dict[char]))
    matrix[0] = header_array
    count = 0

    while count < rows - 1:
        row_arr = []
        row_no = count + 1
        row = Ci[count * len(second_key):(count + 1) * len(second_key)]
        print("Row ", row)
        if len(row) < len(second_key):
            index = len(row)
            while index < len(second_key):
                row = row + 'X'
                index = index + 1
        for char in row:
            row_arr.append(char)
        count = count + 1
        print(row)
        # matrix = np.append(matrix,np.array([row]),axis=0)
        matrix[row_no] = row_arr
    print(matrix)
    return matrix


def get_final_encrpytion(matrix, alphabet_dict):
    encrypted_text = ""
    df = pd.DataFrame.from_records(matrix)
    df = df.sort_values(by=df.index[0], ascending=False, axis=1)
    print("Column", df)
    for column in df:
        charcol = df[column]
        for i, char in enumerate(charcol):
            if i != 0:
                encrypted_text = encrypted_text + str(char)
    return encrypted_text


def get_dedcryption_matrix_str(encrypted_final, second_key, alphabet_dict):
    rows = math.ceil(len(encrypted_final) / len(second_key)) + 1
    cols = len(second_key)
    print("Matrix rows ", rows)
    print("Matrix cols", cols)
    matrix = np.zeros((rows, cols), dtype=object)
    # matrix = [len(Ci)//len(second_key)][len(second_key)]

    header_array = []
    for char in second_key:
        header_array.append(int(alphabet_dict[char]))

    header_array = [sorted(header_array).index(x) for x in header_array]

    list_rows = []
    # Break down final encrypted string
    break_size = rows - 1
    header_array.sort(reverse=True)

    for i in header_array:
        list = []
        list.append(i)

        list_rows.append(list)
    print(list_rows)

    list_rows_chars = [encrypted_final[i:i + break_size] for i in range(0, len(encrypted_final), break_size)]

    for i, char_group in enumerate(list_rows_chars):
        list_rows[i].append(char_group)

    print("Decrpytion Matrix  Shape ", list_rows)
    header_array_index = []
    for char in second_key:
        header_array_index.append(int(alphabet_dict[char]))

    header_array_index = [sorted(header_array_index).index(x) for x in header_array_index]
    print(header_array_index)

    return_str = ""
    for i, index in enumerate(header_array_index):
        for j, row in enumerate(list_rows):
            if index == row[0]:
                return_str = return_str + row[1][0]
    for i, index in enumerate(header_array_index):
        for j, row in enumerate(list_rows):
            if index == row[0]:
                return_str = return_str + row[1][1]
    print("Return str " + return_str)

    return return_str


def get_decryption_str_array(decrpytion_matrix_str, alphabet_dict, keyword):
    arr_list = []
    for char in decrpytion_matrix_str:
        arr_list.append(alphabet_dict[char])

    keyword_pad = generate_keyword_pad(decrpytion_matrix_str, keyword)

    keyword_pad_arr = []

    for char in keyword_pad:
        keyword_pad_arr.append(alphabet_dict[char])

    print("Keypad", keyword_pad_arr)

    raw_index_list = [x1 - x2 for (x1, x2) in zip(arr_list, keyword_pad_arr)]
    raw_index_list = [x if x >= 0 else x + 26 for x in raw_index_list]
    print(raw_index_list)
    return raw_index_list


def get_final_decryption(decrpytion_str_array, alphabet_dict):
    decrypted_str = ""
    for char in decrpytion_str_array:
        char = char % 26
        decrypted_str = decrypted_str + str([k for k, v in alphabet_dict.items() if v == char][0])

    return decrypted_str[::-1]


if __name__ == '__main__':
    main()
