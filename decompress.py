from tools import *


def decompress():
    filename, Y, block_width, block_height, img_width, img_height = read_to_decompress()

    with open('matrix_2.txt', 'r') as file:
        w_matrix_2 = file.readlines()
    w_matrix_2 = [[float(n) for n in x.split()] for x in w_matrix_2]

    X_end = transform_xend(Y, w_matrix_2)

    result_matrix = [[0 for j in range(img_height)] for i in range(img_width)]
    X_end = copy.deepcopy(X_end)

    result_matrix = to_pixels(result_matrix, X_end, block_width, block_height)
    draw_from_network(result_matrix, filename[:-4] + '.jpg', folder='result')