from PIL import Image, ImageDraw
from math import sqrt
import copy
import os
from tqdm import tqdm
import random
import numpy as np

# ввод файла и начальных параметров
def init():
    # считывание файлов
    print('Enter filename: ')
    for filename in os.listdir("img"):
        if filename != '.ipynb_checkpoints':
            print('- ', filename)

    filename = input()

    image = Image.open('img/' + filename)  # открываем изображение
    img_width = image.size[0]  # определяем ширину
    img_height = image.size[1]  # определяем высоту
    pix_matrix = image.load()  # выгружаем значения пикселей
    print('Filename size: ', img_width, 'x', img_height)

    return image, img_width, img_height, pix_matrix, filename

def read_to_decompress():
    print('Enter filename: ')
    for filename in os.listdir("middle"):
        if filename != '.ipynb_checkpoints':
            print('- ', filename)

    filename = input()  # выбор файла

    load = np.load(f'middle/{filename[:-4]}.npz', allow_pickle=True)

    Y = load['Y_end']

    block_width = load['parametrs'][0]
    block_height = load['parametrs'][1]
    img_width = load['parametrs'][2]
    img_height = load['parametrs'][3]

    return filename, Y, block_width, block_height, img_width, img_height

def transform_xend(Y, w_matrix_2):
    X_end = []
    for i in tqdm(range(len(Y))):
        X = multi_matrix(Y[i], w_matrix_2)
        X_end.append(X)

    for i in range(len(X_end)):
        X_end[i] = X_end[i][0]

    for i in range(len(X_end)):
        X_end[i] = [X_end[i][j:j + 3] for j in range(0, len(X_end[i]), 3)]

    return X_end

# перевод пикселей в шкалу от -1 до 1
def scale(img_width, img_height, pix_matrix):
    rgb_matrix = []
    for x in tqdm(range(img_width)):
        rgb_row = []
        for y in range(img_height):
            rgb_pix = []
            r = pix_matrix[x, y][0]
            g = pix_matrix[x, y][1]
            b = pix_matrix[x, y][2]

            rgb_pix = [(2 * r / 255) - 1, (2 * g / 255) - 1, (2 * b / 255) - 1]
            rgb_row.append(rgb_pix)

        rgb_matrix.append(rgb_row)
    return rgb_matrix

def make_blocks(rgb_matrix, img_width, img_height):
    # ввод размера блоков
    print('Please, enter size of blocks: ')
    block_width = int(input('width: '))
    if block_width > img_width:
        print('Too much value. Please, try again')
        block_width = int(input('width: '))

    block_height = int(input('height: '))
    if block_height > img_height:
        print('Too much value. Please, try again')
        block_height = int(input('height: '))

    blocks = []
    i = 0
    j = 0
    mirror_iter = 1

    # разбиение на блоки
    if len(rgb_matrix) % block_width != 0:
        while len(rgb_matrix) % block_width != 0:
            rgb_matrix.append(rgb_matrix[-mirror_iter])
            mirror_iter += 1

    if len(rgb_matrix[0]) % block_height != 0:
        while len(rgb_matrix[0]) % block_height != 0:
            if len(rgb_matrix[0]) % block_height == 0:
                break
            for item in range(len(rgb_matrix)):
                rgb_matrix[item].append(rgb_matrix[item][-1])

    for i in range(0, len(rgb_matrix), block_width):
        for j in range(0, len(rgb_matrix[0]), block_height):
            line = []
            for x in range(block_width):
                for y in range(block_height):
                    line += rgb_matrix[i + x][j + y]
            blocks.append(line)

    return blocks, block_width, block_height

def train_network(first_layer_neuron, blocks, rgb_matrix, block_width, block_height, filename, error):
    # матрицы весов 1-го слоя
    w_matrix_1 = [[random.uniform(-0.1, 0.1) for j in range(first_layer_neuron)] for i in range(len(blocks[0]))]
    w_matrix_2 = transpose(w_matrix_1)
    iteration = 1
    # нейросеть
    while True:
        sum_err = 0
        for i in tqdm(range(len(blocks))):
            # функции сети
            Y = multi_matrix([blocks[i]], w_matrix_1)
            X = multi_matrix(Y, w_matrix_2)
            dX = dev_matrix(X, [blocks[i]])

            # обучение
            w_matrix_before = w_matrix_2
            w_matrix_2 = correct_w_matrix(w_matrix_1, w_matrix_2, Y, dX, layer=2, alphaY=0.01)
            w_matrix_1 = correct_w_matrix(w_matrix_1, w_matrix_before, Y, dX, X=X, alphaX=0.01)
            w_matrix_1 = norm_w_matrix(w_matrix_1)
            w_matrix_2 = norm_w_matrix(w_matrix_2)
            sum_err += sqrt_error(dX)

        print(f'\nError {iteration}: ', sum_err)
        iteration += 1

        # запись промежуточного или конечного результата
        if sum_err < float(error):
            write_result(blocks, w_matrix_1, w_matrix_2, rgb_matrix, block_width, block_height, filename)
            break

def write_result(blocks, w_matrix_1, w_matrix_2, rgb_matrix, block_width, block_height, filename):
    X_end = []
    for i in tqdm(range(len(blocks))):
        # функции сети
        Y = multi_matrix([blocks[i]], w_matrix_1)
        X = multi_matrix(Y, w_matrix_2)
        X_end.append(X)

    for i in range(len(X_end)):
        X_end[i] = X_end[i][0]

    for i in range(len(X_end)):
        X_end[i] = [X_end[i][j:j + 3] for j in range(0, len(X_end[i]), 3)]

    result_matrix = copy.deepcopy(rgb_matrix)
    X_end = copy.deepcopy(X_end)

    result_matrix = to_pixels(result_matrix, X_end, block_width, block_height)
    draw_from_network(result_matrix, filename, folder='result')

    # сохранение весов
    with open('matrix_1.txt', 'w') as f:
        for i in range(len(w_matrix_1)):
            for j in range(len(w_matrix_1[0])):
                f.write(f"{w_matrix_1[i][j]} ")
            f.write('\n')

    with open('matrix_2.txt', 'w') as f:
        for i in range(len(w_matrix_2)):
            for j in range(len(w_matrix_2[0])):
                f.write(f"{w_matrix_2[i][j]} ")
            f.write('\n')

def transpose(matrix):
    matrix = copy.deepcopy(matrix)
    trans_matrix = [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    return trans_matrix

def multi_matrix(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    zip_b = zip(*b)
    zip_b = list(zip_b)  # массив столбцов второй матрицы

    return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in zip_b] for row_a in a]

    # zip(row_a, col_b) - массив из пар элемента строки первой матрицы и элемента столбца второй матрицы
    # перемножаем эту пару, складываем

def dev_matrix(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    new_matrix = [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return new_matrix

def draw_from_network(rgb_matrix, filename, folder=''):
    img_width = len(rgb_matrix)
    img_height = len(rgb_matrix[0])
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)
    for x in range(img_width):
        for y in range(img_height):
            draw.point((x, y), (int(255 * (rgb_matrix[x][y][0] + 1) / 2), int(255 * (rgb_matrix[x][y][1] + 1) / 2),
                                int(255 * (rgb_matrix[x][y][2] + 1) / 2)))
    image.save(f'{folder}/' + filename, "JPEG")

def to_pixels(result_matrix, X, block_width, block_height):
    n=0
    for i in range(0, len(result_matrix), block_width):
        for j in range(0, len(result_matrix[0]), block_height):
            m = 0
            for x in range(block_width):
                for y in range(block_height):
                    result_matrix[i + x][j + y] = []
                    result_matrix[i + x][j + y] += X[n][m]
                    m += 1
            n += 1
    return result_matrix

def sum_matrix(a, b):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    new_matrix = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return new_matrix

def alpha(matrix):
    matrix = copy.deepcopy(matrix)
    alpha = 1 / multi_matrix(matrix, transpose(matrix))[0][0]
    return alpha

def vector_module(w_vector):
    w_vector = copy.deepcopy(w_vector)
    return sqrt(sum([i ** 2 for i in w_vector]))

def correct_w_matrix(w_matrix_1, w_matrix_2, Y, dX, X=[], layer=1, alphaX = 0.005, alphaY=0.005):
    w_matrix_1 = copy.deepcopy(w_matrix_1)
    w_matrix_2 = copy.deepcopy(w_matrix_2)
    Y = copy.deepcopy(Y)
    X = copy.deepcopy(X)
    dX = copy.deepcopy(dX)

    if layer == 1:
        before_matrix = multi_matrix(transpose(X), dX)
        before_matrix = multi_matrix(before_matrix, transpose(w_matrix_2))
        devision_matrix = [[alphaX * before_matrix[i][j] for j in range(len(before_matrix[0]))] for i in range(len(before_matrix))]

        return dev_matrix(w_matrix_1, devision_matrix)
    else:
        before_matrix = multi_matrix(transpose(Y), dX)
        devision_matrix = [[alphaY * before_matrix[i][j] for j in range(len(before_matrix[0]))] for i in range(len(before_matrix))]

        return dev_matrix(w_matrix_2, devision_matrix)

def norm_w_matrix(w_matrix):
    w_matrix = copy.deepcopy(w_matrix)
    w_matrix_norm = [[w_matrix[i][j]/vector_module(w_matrix[i]) for j in range(len(w_matrix[0]))] for i in range(len(w_matrix))]
    return w_matrix_norm

def sqrt_error(dX):
    dX = copy.deepcopy(dX)
    dx_sqrt = list(map(lambda x: x ** 2, dX[0]))
    dx_sqrt = sum(dx_sqrt)
    return dx_sqrt