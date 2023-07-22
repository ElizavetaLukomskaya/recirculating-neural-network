from training import train
from compress import compress
from decompress import decompress


while True:
    print("----------MENU----------")
    print("1) Training of neural network\n2) Compress image\n3) Decompress image\n0) Exit\n")
    menu = input("Enter number: ")

    if menu == '1':
        err = input("Error: ")
        train(error=err)

    elif menu == '2':
        compress()

    elif menu == '3':
        decompress()

    elif menu == '0':
        exit()