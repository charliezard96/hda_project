import numpy as np
import matplotlib.pyplot as plt

def extract_history(stringa):

    f = open(stringa, 'r')
    temp = []
    for line in f:
        line = line[1:-1]
        temp = line.split(',')
    f.close()

    content = []
    for elem in temp:
        content.append(float(elem))

    return content

def main():

    a = extract_history('history\\GRIDSEARCH\\Autoencoder\\historyAutoencoderModel.txt')
    b = extract_history('history\\history fast on validation 4 fun\\historyAutoencoderModel_withBatch.txt')

    plt.plot(a)
    plt.plot(b)
    plt.show()

    z=1

if __name__ == "__main__":
    main()