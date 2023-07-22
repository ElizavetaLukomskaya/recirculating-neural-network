from tools import *

def train(error=1000):
    image, img_width, img_height, pix_matrix, filename = init()
    rgb_matrix = scale(img_width, img_height, pix_matrix)
    blocks, block_width, block_height = make_blocks(rgb_matrix, img_width, img_height)

    print(f'In your block there are {len(blocks[0])} input neurons.')
    first_layer_neuron = int(input('Enter count of neurons in the first layer: '))

    train_network(first_layer_neuron, blocks, rgb_matrix, block_width, block_height, filename, error)

