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

    a = extract_history('history\\CLASSIFIER\\ABN_1024_1024_300epochs\\loss_history_classifier_ABN_1024_1024_epochs300.txt')
    b = extract_history('history\\CLASSIFIER\\ABN_1024_1024_300epochs\\val_acc_history_classifier_ABN_1024_1024_epochs300.txt')

    plt.plot(a)
    #plt.plot(b)
    plt.grid()
    plt.show()

    z=1

if __name__ == "__main__":
    main()