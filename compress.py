from tools import *

def compress():
    image, img_width, img_height, pix_matrix, filename = init()
    rgb_matrix = scale(img_width, img_height, pix_matrix)
    blocks, block_width, block_height = make_blocks(rgb_matrix, img_width, img_height)

    print(f'In your block there are {len(blocks[0])} input neurons.')
    first_layer_neuron = int(input('Enter count of neurons in the first layer: '))

    with open('matrix_1.txt', 'r') as file:
        w_matrix_1 = file.readlines()
    w_matrix_1 = [[float(n) for n in x.split()] for x in w_matrix_1]

    Y_end = []
    for i in tqdm(range(len(blocks))):
        Y = multi_matrix([blocks[i]], w_matrix_1)
        Y_end.append(Y)

    parametrs = [block_width, block_height, img_width, img_height]

    np.savez_compressed(f'middle/{filename[:-4]}', parametrs=parametrs, Y_end=Y_end)